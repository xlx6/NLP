from transformers import TrainerCallback
import math
import swanlab

class AlphaScheduler(TrainerCallback):
    def __init__(self, total_steps, warmup_ratio=0.1):
        self.total_steps = total_steps
        self.warmup_steps = int(total_steps * warmup_ratio)
        self.non_warmup_steps = max(1, total_steps - self.warmup_steps)
        self.v_midpoint = self.warmup_steps + self.non_warmup_steps // 2
        self.v_descent_steps = max(1, self.v_midpoint - self.warmup_steps)
        self.v_ascent_steps = max(1, self.total_steps - self.v_midpoint)

    def on_step_end(self, args, state, control, **kwargs):
        step = state.global_step
        trainer = kwargs['trainer']

        if step > self.total_steps:
            return

        scheduler_type = args.hp_method.lower()
        assert scheduler_type in ['cosine_growth', 'cosine_vshape', 'cosine_descent', 'cosine_^shape']

        if scheduler_type == 'cosine_growth':
            new_alpha = self._cosine_growth_schedule(step)
        elif scheduler_type == 'cosine_vshape':
            new_alpha = self._vshape_schedule(step)
        elif scheduler_type == 'cosin_descent':
            new_alpha = self._cosine_descent_schedule(step)
        else:
            new_alpha = self._cosine_growth_descent_schedule(step)

        trainer.alpha = new_alpha
        if step % 10 == 0:
            swanlab.log({"alpha": new_alpha})

    def _cosine_growth_schedule(self, step):
        if step <= self.warmup_steps:
            return 0.0
        else:
            progress = (step - self.warmup_steps) / self.non_warmup_steps
            return 1 - math.cos(progress * (math.pi / 2))

    def _vshape_schedule(self, step):
        if step <= self.warmup_steps:
            return 1.0
        elif step <= self.v_midpoint:
            progress = (step - self.warmup_steps) / self.v_descent_steps
            return math.cos(progress * (math.pi / 2))
        else:
            progress = (step - self.v_midpoint) / self.v_ascent_steps
            return math.sin(progress * (math.pi / 2))

    def _cosine_descent_schedule(self, step):
        if step <= self.warmup_steps:
            return 1.0
        else:
            progress = (step - self.warmup_steps) / self.non_warmup_steps
            return math.cos(progress * (math.pi / 2))

    def _cosine_growth_descent_schedule(self, step):
        """alpha: 0->1->0"""
        if step <= self.warmup_steps:
            return 0.0
        elif step <= self.v_midpoint:
            progress = (step - self.warmup_steps) / self.v_descent_steps
            return 1 - math.cos(progress * (math.pi / 2))
        else:
            progress = (step - self.v_midpoint) / self.v_ascent_steps
            return 1 - math.sin(progress * (math.pi / 2))
