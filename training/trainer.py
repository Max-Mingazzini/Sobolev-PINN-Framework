import time
import torch
import torch.nn.functional as F
from torch.optim import Adam
from utils.derivatives import J_t_model, J_p_model, J_tk_model


class Trainer:
    def __init__(self, model, ode, cfg, optimizer_cls=Adam):
        # device and model
        self.device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device)
        self.ode = ode

        train_cfg = cfg.get("train", cfg.get("training", {}))
        lr = float(train_cfg.get("learning_rate", 1e-3))
        wd = float(train_cfg.get("weight_decay", 0.0))
        self.opt = optimizer_cls(self.model.parameters(), lr=lr, weight_decay=wd)

        # loss weights
        self.w_phys = float(train_cfg.get("w_phys", 1.0))
        self.w_data = float(train_cfg.get("w_data", 0.0))
        self.w_time = float(train_cfg.get("w_time", 0.0))
        self.w_param = float(train_cfg.get("w_param", 0.0))
        self.w_ic = float(train_cfg.get("w_ic", 1.0))

        # Temporal Sobolev
        self.time_order = int(train_cfg.get("time_order", 0))
        self.time_order = max(0, min(self.time_order, 2))
        self.time_order_weights = train_cfg.get("time_order_weights", None)

        # misc
        self.clip_grad = float(train_cfg.get("clip_grad", 0.0))
        self.epochs = int(train_cfg.get("epochs", 500))
        self.log_interval = int(train_cfg.get("log_interval", 50))
        self.history = []

        # IC info
        data_cfg = cfg.get("data", {})
        self.ic_t0 = float(data_cfg.get("t0", 0.0))
        y0 = torch.as_tensor(data_cfg.get("initial_condition", []), dtype=torch.get_default_dtype())
        if y0.ndim == 0:
            y0 = y0.view(1)
        self.ic_y0 = y0.view(1, -1)  # [1,S]

    def _to_device(self, batch):
        out = {}
        for k, v in batch.items():
            out[k] = v.to(self.device) if isinstance(v, torch.Tensor) else v
        return out

    def train(self, loader, val_loader=None):
        """
        Defines training using the loss terms for:
        - Data
        - Physics
        - Initial Condition
        - Temporal Sobolev loss (FO and SO)
        - Parameter Sobolev loss
        """
        total_start_time = time.perf_counter()
        self.model.train()
        self.history.clear()

        for epoch in range(1, self.epochs + 1):
            epoch_start_time = time.perf_counter()
            tracker = {
                "loss": 0.0,
                "data": 0.0,
                "phys": 0.0,
                "time": 0.0,
                "param": 0.0,
                "ic": 0.0,
            }
            n = 0

            for batch in loader:
                batch = self._to_device(batch)
                t = batch["t"]  # [B,1]
                p = batch["p"]  # [B,P]
                y_r = batch["y"]  # [B,S]
                tt = batch.get("time_targets", None)  # [B,K,S] or None
                S_r = batch.get("sens", None)  # [B,S,P] or None

                # Forward
                y_hat = self.model(t, p)  # [B,S]

                # Physics loss
                L_phys = torch.tensor(0.0, device=self.device)
                if self.w_phys > 0:
                    Jt_hat = J_t_model(self.model, t, p)  # [B,S]
                    rhs = self.ode.f(t, y_hat, p)  # [B,S]
                    L_phys = F.mse_loss(Jt_hat, rhs)

                # Data loss
                L_data = torch.tensor(0.0, device=self.device)
                if self.w_data > 0:
                    L_data = F.mse_loss(y_hat, y_r)

                # Temporal Sobolev
                L_time = torch.tensor(0.0, device=self.device)
                if self.w_time > 0 and tt is not None:
                    K_avail = tt.shape[1]
                    use_K = min(self.time_order, K_avail)
                    if use_K > 0:
                        weights = self.time_order_weights[:use_K] if self.time_order_weights else [1.0] * use_K
                        denom = max(sum(weights), 1e-12)

                        # k = 1
                        Jt_hat2 = Jt_hat if self.w_phys > 0 else J_t_model(self.model, t, p)
                        L_time1 = F.mse_loss(Jt_hat2, tt[:, 0, :])
                        L_time = L_time + weights[0] * L_time1

                        # k = 2
                        if use_K >= 2:
                            Jtt_hat = J_tk_model(self.model, t, p, order=2)
                            L_time2 = F.mse_loss(Jtt_hat, tt[:, 1, :])
                            L_time = L_time + weights[1] * L_time2

                        L_time = L_time / denom

                # Parameter sensitivities
                L_param = torch.tensor(0.0, device=self.device)
                if self.w_param > 0 and S_r is not None:
                    Jp_hat = J_p_model(self.model, t, p)
                    L_param = F.mse_loss(Jp_hat, S_r)

                # IC loss on the t==t0 samples in this batch
                L_ic = torch.tensor(0.0, device=self.device)
                if self.w_ic > 0.0:
                    # t is [B,1]; match exactly t0 (grid includes endpoints).
                    mask = (t.squeeze(-1) - self.ic_t0).abs() < 1e-12
                    if mask.any():
                        y_hat_ic = y_hat[mask]  # [B0,S]
                        y0_rep = self.ic_y0.to(self.device).expand_as(y_hat_ic)
                        L_ic = F.mse_loss(y_hat_ic, y0_rep)

                # Total
                loss = (
                    self.w_phys * L_phys
                    + self.w_data * L_data
                    + self.w_time * L_time
                    + self.w_param * L_param
                    + self.w_ic * L_ic
                )

                # Step
                self.opt.zero_grad()
                loss.backward()
                if self.clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
                self.opt.step()

                # Track
                tracker["loss"] += float(loss)
                tracker["data"] += float(L_data)
                tracker["phys"] += float(L_phys)
                tracker["time"] += float(L_time)
                tracker["param"] += float(L_param)
                tracker["ic"] += float(L_ic)
                n += 1

            avg = {k: v / max(1, n) for k, v in tracker.items()}
            val = None

            epoch_duration = float(time.perf_counter() - epoch_start_time)  # validation ignored for time

            if epoch % self.log_interval == 0 or epoch in (1, self.epochs):
                print(
                    f"Epoch {epoch} | duration {epoch_duration:.1f}s | "
                    f"loss {avg['loss']:.4e} | data {avg['data']:.2e} "
                    f"phys {avg['phys']:.2e} time {avg['time']:.2e} "
                    f"param {avg['param']:.2e} ic {avg['ic']:.2e}"
                )

            if val_loader is not None and (epoch % self.log_interval == 0 or epoch == self.epochs):
                val = self.evaluate(val_loader)

            if val is not None:
                row = {
                    "epoch": epoch,
                    "epoch_duration": epoch_duration,
                    **avg,
                    **{f"val_{k}": v for k, v in val.items()},
                }
            else:
                row = {"epoch": epoch, "epoch_duration": epoch_duration, **avg}
            self.history.append(row)
        self.total_train_time = time.perf_counter() - total_start_time
        return self

    @torch.no_grad()
    def evaluate(self, loader, return_dict=False):
        state = self.model.training
        self.model.eval()
        sums = {"data": 0.0, "phys": 0.0, "dt": 0.0, "sens": 0.0, "ic": 0.0}
        counts = {"data": 0, "phys": 0, "dt": 0, "sens": 0, "ic": 0}
        for b in loader:
            b = self._to_device(b)
            t, p, y = b["t"], b["p"], b["y"]
            y_hat = self.model(t, p)

            # data
            sums["data"] += float(F.mse_loss(y_hat, y, reduction="sum"))
            counts["data"] += y.numel()

            # physics
            Jt_hat = J_t_model(self.model, t, p)
            rhs = self.ode.f(t, y_hat, p)
            diff = Jt_hat - rhs
            sums["phys"] += float(torch.sum(diff * diff))
            counts["phys"] += diff.numel()

            # time Sobolev (k=1) if present
            if b.get("time_targets") is not None:
                d1 = b["time_targets"][:, 0, :]
                err = Jt_hat - d1
                sums["dt"] += float(torch.sum(err * err))
                counts["dt"] += err.numel()

            # parameter sensitivities if present
            if b.get("sens") is not None:
                Jp_hat = J_p_model(self.model, t, p)
                err = Jp_hat - b["sens"]
                sums["sens"] += float(torch.sum(err * err))
                counts["sens"] += err.numel()

            # IC metric on t==t0 points present in this batch
            mask = (t.squeeze(-1) - self.ic_t0).abs() < 1e-12
            if mask.any():
                y0_rep = self.ic_y0.to(self.device).expand_as(y_hat[mask])
                err = y_hat[mask] - y0_rep
                sums["ic"] += float(torch.sum(err * err))
                counts["ic"] += err.numel()

        avg = {k: (sums[k] / max(1, counts[k])) for k in sums}
        if not return_dict:
            print(
                f"  [val] data {avg['data']:.3e} | phys {avg['phys']:.3e} "
                f"| dt {avg['dt']:.3e} | sens {avg['sens']:.3e} | ic {avg['ic']:.3e}"
            )
        self.model.train(state)
        return avg
