# Copyright 2022 Cerebras Systems.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This module contains the callback class that logs all metrics attached to the model."""

from cerebras.modelzoo.trainer.callbacks import Callback
from cerebras.pytorch.metrics.metric import Metric


class ModelEvalMetrics(Callback):
    """Callback class that logs all metrics attached to the model."""

    def on_validate_end(self, trainer, model, loop):
        print("\n" + "="*80)
        print("[DEBUG] 15. CALLBACK: ModelEvalMetrics.on_validate_end()")
        print("="*80)
        print(f"[DEBUG] 15.1 Collecting metrics from model...")

        if trainer.backend.is_e2e_execution:
            # Print all metrics that are attached to the model
            metrics = {}

            print(f"[DEBUG] 15.2 Iterating through model.modules() to find Metric objects...")
            for metric in model.modules():
                if isinstance(metric, Metric):
                    if metric.num_updates == 0:
                        print(f"[DEBUG] 15.3 WARNING: Metric '{metric.name}' has 0 updates - skipping")
                        trainer.logger.warning(
                            f"Skipping logging unused metric `{metric.name}` "
                            f"To remove this warning, either remove it from "
                            f"the model or step the metric every step."
                        )
                    else:
                        print(f"[DEBUG] 15.4 *** TRANSFERRING METRIC '{metric.name}' FROM CSX TO CPU ***")
                        print(f"[DEBUG] 15.5   Calling float(metric) which triggers .item() -> CSX→CPU transfer")
                        metrics[metric.name] = float(metric)
                        print(f"[DEBUG] 15.6   Value received on CPU: {metrics[metric.name]}")
                        metric.reset()

            print(f"[DEBUG] 15.7 All metrics transferred to CPU!")
            print(f"[DEBUG] 15.8 Logging metrics:")
            trainer.logger.info("Evaluation metrics:")
            for name, metric in metrics.items():
                trainer.logger.info(f"  - {name} = {float(metric)}")
                print(f"[DEBUG] 15.9   {name} = {float(metric)}")

            trainer.log_metrics(**metrics)
            print(f"[DEBUG] 15.10 Metrics logged!\n")

    def on_save_trainer_state(self, trainer, state_dict):
        pass

    def on_load_trainer_state(self, trainer, state_dict):
        pass
