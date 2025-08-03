# Advanced Usage Guide

This guide covers advanced features and optimization techniques for the Segmentation Robustness Framework.

## 📚 Table of Contents

1. [Performance Optimization](#performance-optimization)
2. [Memory Management](#memory-management)
3. [Distributed Evaluation](#distributed-evaluation)
4. [Custom Evaluation Strategies](#custom-evaluation-strategies)
5. [Advanced Visualization](#advanced-visualization)
6. [Integration with Other Frameworks](#integration-with-other-frameworks)
7. [Debugging and Profiling](#debugging-and-profiling)
8. [Production Deployment](#production-deployment)

## 🚀 Performance Optimization

### GPU Optimization

#### Mixed Precision Training

```python
import torch
from torch.cuda.amp import autocast, GradScaler

# Enable mixed precision for faster computation
def evaluate_with_mixed_precision(pipeline, model, dataset):
    scaler = GradScaler()
    
    with autocast():
        # Forward pass with automatic mixed precision
        outputs = model(images)
        loss = criterion(outputs, targets)
    
    # Backward pass with gradient scaling
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    
    return outputs

# Use in pipeline
pipeline = SegmentationRobustnessPipeline(
    model=model,
    dataset=dataset,
    attacks=attacks,
    metrics=metrics,
    batch_size=8,
    device="cuda"
)

# Custom evaluation with mixed precision
results = pipeline.run_with_custom_evaluation(evaluate_with_mixed_precision)
```

#### CUDA Optimizations

```python
# Optimize CUDA performance
import torch

# Set memory fraction to avoid OOM
torch.cuda.set_per_process_memory_fraction(0.8)

# Enable cuDNN benchmarking for fixed input sizes
torch.backends.cudnn.benchmark = True

# Use deterministic algorithms for reproducibility
torch.backends.cudnn.deterministic = True

# Optimize for inference
model.eval()
with torch.no_grad():
    outputs = model(images)
```

#### Batch Size Optimization

```python
def find_optimal_batch_size(model, dataset, device="cuda"):
    """Find optimal batch size for given model and device."""
    batch_sizes = [1, 2, 4, 8, 16, 32]
    optimal_batch_size = 1
    
    for batch_size in batch_sizes:
        try:
            # Test with sample data
            sample_data = torch.randn(batch_size, 3, 512, 512).to(device)
            
            # Warm up
            for _ in range(3):
                with torch.no_grad():
                    _ = model(sample_data)
            
            # Measure time
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            
            start_time.record()
            with torch.no_grad():
                _ = model(sample_data)
            end_time.record()
            
            torch.cuda.synchronize()
            elapsed_time = start_time.elapsed_time(end_time)
            
            print(f"Batch size {batch_size}: {elapsed_time:.2f} ms")
            optimal_batch_size = batch_size
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"Batch size {batch_size}: OOM")
                break
    
    return optimal_batch_size

# Use optimal batch size
optimal_batch_size = find_optimal_batch_size(model, dataset)
pipeline = SegmentationRobustnessPipeline(
    model=model,
    dataset=dataset,
    attacks=attacks,
    metrics=metrics,
    batch_size=optimal_batch_size
)
```

### Data Loading Optimization

#### Custom DataLoader

```python
from torch.utils.data import DataLoader
import torch.multiprocessing as mp

class OptimizedDataLoader:
    """Optimized data loader for better performance."""
    
    def __init__(self, dataset, batch_size=4, num_workers=4, pin_memory=True):
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=2 if num_workers > 0 else None
        )
    
    def __iter__(self):
        return iter(self.dataloader)

# Use optimized data loader
optimized_loader = OptimizedDataLoader(dataset, batch_size=8, num_workers=4)
```

#### Dataset Caching

```python
class CachedDataset(torch.utils.data.Dataset):
    """Dataset with caching for faster loading."""
    
    def __init__(self, dataset, cache_size=100):
        self.dataset = dataset
        self.cache = {}
        self.cache_size = cache_size
        self.cache_order = []
    
    def __getitem__(self, idx):
        if idx in self.cache:
            # Move to end of cache order (LRU)
            self.cache_order.remove(idx)
            self.cache_order.append(idx)
            return self.cache[idx]
        
        # Load from original dataset
        item = self.dataset[idx]
        
        # Cache if not full
        if len(self.cache) < self.cache_size:
            self.cache[idx] = item
            self.cache_order.append(idx)
        else:
            # Remove least recently used
            lru_idx = self.cache_order.pop(0)
            del self.cache[lru_idx]
            self.cache[idx] = item
            self.cache_order.append(idx)
        
        return item
    
    def __len__(self):
        return len(self.dataset)

# Use cached dataset
cached_dataset = CachedDataset(original_dataset, cache_size=200)
```

## 💾 Memory Management

### Memory-Efficient Evaluation

```python
class MemoryEfficientPipeline(SegmentationRobustnessPipeline):
    """Memory-efficient pipeline with automatic memory management."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.memory_threshold = 0.8  # 80% memory usage threshold
    
    def _check_memory_usage(self):
        """Check GPU memory usage."""
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated()
            memory_reserved = torch.cuda.memory_reserved()
            memory_total = torch.cuda.get_device_properties(0).total_memory
            
            usage_ratio = memory_allocated / memory_total
            return usage_ratio > self.memory_threshold
        return False
    
    def _clear_memory(self):
        """Clear GPU memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def evaluate_clean(self):
        """Memory-efficient clean evaluation."""
        results = {}
        
        for batch_idx, (images, targets) in enumerate(self.dataloader):
            # Check memory usage
            if self._check_memory_usage():
                self._clear_memory()
            
            # Process batch
            with torch.no_grad():
                outputs = self.model(images.to(self.device))
                predictions = torch.argmax(outputs, dim=1)
            
            # Compute metrics
            for metric in self.metrics:
                metric_value = metric(targets, predictions.cpu())
                metric_name = self.metric_names[self.metrics.index(metric)]
                
                if metric_name not in results:
                    results[metric_name] = []
                results[metric_name].append(metric_value)
            
            # Clear intermediate tensors
            del outputs, predictions
            if self._check_memory_usage():
                self._clear_memory()
        
        return results

# Use memory-efficient pipeline
pipeline = MemoryEfficientPipeline(
    model=model,
    dataset=dataset,
    attacks=attacks,
    metrics=metrics,
    batch_size=2  # Smaller batch size
)
```

### Gradient Checkpointing

```python
class GradientCheckpointedModel(torch.nn.Module):
    """Model with gradient checkpointing for memory efficiency."""
    
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
    
    def forward(self, x):
        return torch.utils.checkpoint.checkpoint(self.base_model, x)

# Use gradient checkpointing
original_model = load_model()
checkpointed_model = GradientCheckpointedModel(original_model)
```

### Dynamic Batch Sizing

```python
class DynamicBatchPipeline(SegmentationRobustnessPipeline):
    """Pipeline with dynamic batch sizing based on memory."""
    
    def __init__(self, *args, initial_batch_size=8, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_batch_size = initial_batch_size
        self.current_batch_size = initial_batch_size
    
    def _adjust_batch_size(self, success: bool):
        """Adjust batch size based on success/failure."""
        if success and self.current_batch_size < 32:
            self.current_batch_size = min(32, self.current_batch_size * 2)
        elif not success and self.current_batch_size > 1:
            self.current_batch_size = max(1, self.current_batch_size // 2)
    
    def evaluate_with_dynamic_batch(self):
        """Evaluate with dynamic batch sizing."""
        results = []
        
        for batch_idx, (images, targets) in enumerate(self.dataloader):
            try:
                # Try with current batch size
                batch_images = images[:self.current_batch_size]
                batch_targets = targets[:self.current_batch_size]
                
                with torch.no_grad():
                    outputs = self.model(batch_images.to(self.device))
                    predictions = torch.argmax(outputs, dim=1)
                
                # Success - increase batch size
                self._adjust_batch_size(True)
                results.append(predictions.cpu())
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    # Failure - decrease batch size
                    self._adjust_batch_size(False)
                    torch.cuda.empty_cache()
                    
                    # Retry with smaller batch
                    try:
                        batch_images = images[:self.current_batch_size]
                        batch_targets = targets[:self.current_batch_size]
                        
                        with torch.no_grad():
                            outputs = self.model(batch_images.to(self.device))
                            predictions = torch.argmax(outputs, dim=1)
                        
                        results.append(predictions.cpu())
                    except RuntimeError:
                        # If still fails, process one by one
                        for i in range(len(images)):
                            with torch.no_grad():
                                output = self.model(images[i:i+1].to(self.device))
                                prediction = torch.argmax(output, dim=1)
                                results.append(prediction.cpu())
        
        return torch.cat(results, dim=0)
```

## 🔄 Distributed Evaluation

### Multi-GPU Evaluation

```python
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel

class MultiGPUPipeline(SegmentationRobustnessPipeline):
    """Pipeline for multi-GPU evaluation."""
    
    def __init__(self, *args, gpu_ids=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.gpu_ids = gpu_ids or list(range(torch.cuda.device_count()))
        
        # Wrap model for multi-GPU
        if len(self.gpu_ids) > 1:
            self.model = DataParallel(self.model, device_ids=self.gpu_ids)
    
    def evaluate_clean(self):
        """Multi-GPU clean evaluation."""
        results = {}
        
        # Distribute data across GPUs
        for batch_idx, (images, targets) in enumerate(self.dataloader):
            # Split batch across GPUs
            batch_size_per_gpu = images.size(0) // len(self.gpu_ids)
            
            gpu_results = []
            for gpu_idx in self.gpu_ids:
                start_idx = gpu_idx * batch_size_per_gpu
                end_idx = start_idx + batch_size_per_gpu
                
                gpu_images = images[start_idx:end_idx].to(f"cuda:{gpu_idx}")
                gpu_targets = targets[start_idx:end_idx].to(f"cuda:{gpu_idx}")
                
                with torch.no_grad():
                    gpu_outputs = self.model(gpu_images)
                    gpu_predictions = torch.argmax(gpu_outputs, dim=1)
                
                gpu_results.append(gpu_predictions.cpu())
            
            # Combine results
            predictions = torch.cat(gpu_results, dim=0)
            
            # Compute metrics
            for metric in self.metrics:
                metric_value = metric(targets, predictions)
                metric_name = self.metric_names[self.metrics.index(metric)]
                
                if metric_name not in results:
                    results[metric_name] = []
                results[metric_name].append(metric_value)
        
        return results

# Use multi-GPU pipeline
pipeline = MultiGPUPipeline(
    model=model,
    dataset=dataset,
    attacks=attacks,
    metrics=metrics,
    gpu_ids=[0, 1, 2, 3]
)
```

### Distributed Data Parallel

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_distributed(rank, world_size):
    """Setup distributed training."""
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_distributed():
    """Cleanup distributed training."""
    dist.destroy_process_group()

class DistributedPipeline(SegmentationRobustnessPipeline):
    """Pipeline for distributed evaluation."""
    
    def __init__(self, *args, rank=0, world_size=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.rank = rank
        self.world_size = world_size
        
        if world_size > 1:
            self.model = DDP(self.model, device_ids=[rank])
    
    def evaluate_clean(self):
        """Distributed clean evaluation."""
        results = {}
        
        # Only process subset of data on each rank
        for batch_idx, (images, targets) in enumerate(self.dataloader):
            if batch_idx % self.world_size != self.rank:
                continue
            
            with torch.no_grad():
                outputs = self.model(images.to(self.device))
                predictions = torch.argmax(outputs, dim=1)
            
            # Compute metrics
            for metric in self.metrics:
                metric_value = metric(targets, predictions.cpu())
                metric_name = self.metric_names[self.metrics.index(metric)]
                
                if metric_name not in results:
                    results[metric_name] = []
                results[metric_name].append(metric_value)
        
        # Gather results from all ranks
        if self.world_size > 1:
            gathered_results = [None] * self.world_size
            dist.all_gather_object(gathered_results, results)
            
            # Combine results
            combined_results = {}
            for rank_results in gathered_results:
                for metric_name, values in rank_results.items():
                    if metric_name not in combined_results:
                        combined_results[metric_name] = []
                    combined_results[metric_name].extend(values)
            
            return combined_results
        
        return results

# Use distributed pipeline
def run_distributed_evaluation(rank, world_size):
    setup_distributed(rank, world_size)
    
    pipeline = DistributedPipeline(
        model=model,
        dataset=dataset,
        attacks=attacks,
        metrics=metrics,
        rank=rank,
        world_size=world_size
    )
    
    results = pipeline.evaluate_clean()
    
    if rank == 0:
        print("Combined results:", results)
    
    cleanup_distributed()

# Launch distributed evaluation
import torch.multiprocessing as mp

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(run_distributed_evaluation, args=(world_size,), nprocs=world_size)
```

## 🎯 Custom Evaluation Strategies

### Progressive Evaluation

```python
class ProgressiveEvaluationPipeline(SegmentationRobustnessPipeline):
    """Pipeline with progressive evaluation strategy."""
    
    def __init__(self, *args, sample_sizes=[100, 500, 1000, -1], **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_sizes = sample_sizes
    
    def progressive_evaluate(self):
        """Evaluate progressively with increasing sample sizes."""
        all_results = {}
        
        for sample_size in self.sample_sizes:
            print(f"Evaluating with {sample_size} samples...")
            
            # Create subset dataset
            if sample_size > 0:
                subset_dataset = torch.utils.data.Subset(
                    self.dataset, 
                    range(min(sample_size, len(self.dataset)))
                )
            else:
                subset_dataset = self.dataset
            
            # Create temporary pipeline
            temp_pipeline = SegmentationRobustnessPipeline(
                model=self.model,
                dataset=subset_dataset,
                attacks=self.attacks,
                metrics=self.metrics,
                batch_size=self.batch_size,
                device=self.device
            )
            
            # Run evaluation
            results = temp_pipeline.run(save=False, show=False)
            all_results[f"sample_size_{sample_size}"] = results
            
            # Print intermediate results
            print(f"Results with {sample_size} samples:")
            for attack_name, attack_results in results.items():
                if attack_name != "clean":
                    print(f"  {attack_name}: {attack_results.get('mean_iou', 'N/A')}")
        
        return all_results

# Use progressive evaluation
pipeline = ProgressiveEvaluationPipeline(
    model=model,
    dataset=dataset,
    attacks=attacks,
    metrics=metrics,
    sample_sizes=[100, 500, 1000, -1]
)

progressive_results = pipeline.progressive_evaluate()
```

### Adaptive Evaluation

```python
class AdaptiveEvaluationPipeline(SegmentationRobustnessPipeline):
    """Pipeline with adaptive evaluation based on model performance."""
    
    def __init__(self, *args, performance_threshold=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.performance_threshold = performance_threshold
    
    def adaptive_evaluate(self):
        """Adaptive evaluation based on model performance."""
        # First, evaluate clean performance
        clean_results = self.evaluate_clean()
        clean_iou = clean_results.get("mean_iou", 0)
        
        print(f"Clean IoU: {clean_iou:.3f}")
        
        if clean_iou < self.performance_threshold:
            print("Model performance below threshold, skipping attack evaluation")
            return {"clean": clean_results}
        
        # If performance is good, run attack evaluation
        attack_results = self.evaluate_attack()
        
        return {
            "clean": clean_results,
            **attack_results
        }
    
    def evaluate_attack(self):
        """Evaluate with attacks."""
        results = {}
        
        for attack in self.attacks:
            print(f"Evaluating attack: {attack}")
            
            attack_results = {}
            for batch_idx, (images, targets) in enumerate(self.dataloader):
                # Apply attack
                adv_images = attack.apply(images.to(self.device), targets.to(self.device))
                
                # Evaluate on adversarial images
                with torch.no_grad():
                    outputs = self.model(adv_images)
                    predictions = torch.argmax(outputs, dim=1)
                
                # Compute metrics
                for metric in self.metrics:
                    metric_value = metric(targets, predictions.cpu())
                    metric_name = self.metric_names[self.metrics.index(metric)]
                    
                    if metric_name not in attack_results:
                        attack_results[metric_name] = []
                    attack_results[metric_name].append(metric_value)
            
            # Average results
            for metric_name, values in attack_results.items():
                attack_results[metric_name] = sum(values) / len(values)
            
            results[f"attack_{attack}"] = attack_results
        
        return results

# Use adaptive evaluation
pipeline = AdaptiveEvaluationPipeline(
    model=model,
    dataset=dataset,
    attacks=attacks,
    metrics=metrics,
    performance_threshold=0.7
)

adaptive_results = pipeline.adaptive_evaluate()
```

## 📊 Advanced Visualization

### Interactive Dashboards

```python
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd

class InteractiveVisualizer:
    """Interactive visualization dashboard."""
    
    def __init__(self, results_data):
        self.results = results_data
        self.df = self._prepare_dataframe()
    
    def _prepare_dataframe(self):
        """Prepare results as DataFrame."""
        data = []
        
        for attack_name, attack_results in self.results.items():
            for metric_name, metric_value in attack_results.items():
                if isinstance(metric_value, (int, float)):
                    data.append({
                        'Attack': attack_name,
                        'Metric': metric_name,
                        'Value': metric_value
                    })
        
        return pd.DataFrame(data)
    
    def create_comparison_dashboard(self):
        """Create interactive comparison dashboard."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Mean IoU Comparison', 'Pixel Accuracy Comparison',
                          'Precision Comparison', 'Recall Comparison'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        metrics = ['mean_iou', 'pixel_accuracy', 'precision', 'recall']
        positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
        
        for metric, pos in zip(metrics, positions):
            metric_data = self.df[self.df['Metric'] == metric]
            
            fig.add_trace(
                go.Bar(
                    x=metric_data['Attack'],
                    y=metric_data['Value'],
                    name=metric.replace('_', ' ').title()
                ),
                row=pos[0], col=pos[1]
            )
        
        fig.update_layout(
            title="Segmentation Robustness Evaluation Dashboard",
            height=800,
            showlegend=False
        )
        
        return fig
    
    def create_attack_parameter_analysis(self, attack_name, param_name):
        """Create analysis of attack parameter effects."""
        # Filter data for specific attack
        attack_data = self.df[self.df['Attack'].str.contains(attack_name)]
        
        # Extract parameter values from attack names
        param_values = []
        for attack in attack_data['Attack'].unique():
            # Extract parameter value from attack name
            # e.g., "attack_FGSM_eps_0p008" -> 0.008
            import re
            match = re.search(r'eps_(\d+p\d+)', attack)
            if match:
                param_str = match.group(1).replace('p', '.')
                param_values.append(float(param_str))
            else:
                param_values.append(0)
        
        attack_data['Parameter'] = param_values
        
        # Create scatter plot
        fig = px.scatter(
            attack_data,
            x='Parameter',
            y='Value',
            color='Metric',
            title=f"{attack_name} Parameter Analysis"
        )
        
        return fig
    
    def create_performance_heatmap(self):
        """Create performance heatmap."""
        # Pivot data for heatmap
        pivot_df = self.df.pivot(index='Attack', columns='Metric', values='Value')
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot_df.values,
            x=pivot_df.columns,
            y=pivot_df.index,
            colorscale='RdYlGn',
            zmid=0.5
        ))
        
        fig.update_layout(
            title="Performance Heatmap",
            xaxis_title="Metrics",
            yaxis_title="Attacks"
        )
        
        return fig

# Use interactive visualizer
visualizer = InteractiveVisualizer(results)

# Create dashboard
dashboard = visualizer.create_comparison_dashboard()
dashboard.show()

# Create parameter analysis
param_analysis = visualizer.create_attack_parameter_analysis("FGSM", "eps")
param_analysis.show()

# Create heatmap
heatmap = visualizer.create_performance_heatmap()
heatmap.show()
```

### Real-time Monitoring

```python
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class RealTimeMonitor:
    """Real-time monitoring of evaluation progress."""
    
    def __init__(self):
        self.metrics_history = {
            'clean': [],
            'fgsm': [],
            'pgd': []
        }
        self.timestamps = []
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
        self.fig.suptitle('Real-time Evaluation Progress')
    
    def update_metrics(self, attack_name, metrics):
        """Update metrics for real-time plotting."""
        self.metrics_history[attack_name].append(metrics)
        self.timestamps.append(time.time())
    
    def animate(self, frame):
        """Animate real-time plots."""
        for i, (attack_name, history) in enumerate(self.metrics_history.items()):
            if history:
                ax = self.axes[i // 2, i % 2]
                ax.clear()
                
                # Plot metric history
                for metric_name in ['mean_iou', 'pixel_accuracy']:
                    values = [h.get(metric_name, 0) for h in history]
                    ax.plot(values, label=metric_name)
                
                ax.set_title(f'{attack_name.upper()} Progress')
                ax.set_xlabel('Batch')
                ax.set_ylabel('Metric Value')
                ax.legend()
                ax.grid(True)
    
    def start_monitoring(self):
        """Start real-time monitoring."""
        ani = FuncAnimation(self.fig, self.animate, interval=1000)
        plt.show()

# Use real-time monitoring
monitor = RealTimeMonitor()

# In your evaluation loop
for batch_idx, (images, targets) in enumerate(dataloader):
    # Process batch
    outputs = model(images)
    predictions = torch.argmax(outputs, dim=1)
    
    # Compute metrics
    metrics = {
        'mean_iou': mean_iou_metric(targets, predictions),
        'pixel_accuracy': pixel_accuracy_metric(targets, predictions)
    }
    
    # Update monitor
    monitor.update_metrics('clean', metrics)
```

## 🔗 Integration with Other Frameworks

### Weights & Biases Integration

```python
import wandb

class WandbPipeline(SegmentationRobustnessPipeline):
    """Pipeline with Weights & Biases integration."""
    
    def __init__(self, *args, project_name="segmentation-robustness", **kwargs):
        super().__init__(*args, **kwargs)
        
        # Initialize wandb
        wandb.init(project=project_name)
        
        # Log configuration
        wandb.config.update({
            "model_type": type(self.model).__name__,
            "dataset_type": type(self.dataset).__name__,
            "batch_size": self.batch_size,
            "device": self.device
        })
    
    def run(self, save=True, show=False):
        """Run evaluation with wandb logging."""
        results = super().run(save=save, show=show)
        
        # Log results to wandb
        for attack_name, attack_results in results.items():
            for metric_name, metric_value in attack_results.items():
                if isinstance(metric_value, (int, float)):
                    wandb.log({
                        f"{attack_name}/{metric_name}": metric_value
                    })
        
        # Log confusion matrix
        if 'confusion_matrix' in results:
            wandb.log({
                "confusion_matrix": wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=results['true_labels'],
                    preds=results['predictions']
                )
            })
        
        return results

# Use wandb pipeline
pipeline = WandbPipeline(
    model=model,
    dataset=dataset,
    attacks=attacks,
    metrics=metrics,
    project_name="my-segmentation-project"
)

results = pipeline.run()
```

### TensorBoard Integration

```python
from torch.utils.tensorboard import SummaryWriter

class TensorboardPipeline(SegmentationRobustnessPipeline):
    """Pipeline with TensorBoard integration."""
    
    def __init__(self, *args, log_dir="./runs", **kwargs):
        super().__init__(*args, **kwargs)
        self.writer = SummaryWriter(log_dir)
    
    def run(self, save=True, show=False):
        """Run evaluation with TensorBoard logging."""
        results = super().run(save=save, show=show)
        
        # Log results to TensorBoard
        for attack_name, attack_results in results.items():
            for metric_name, metric_value in attack_results.items():
                if isinstance(metric_value, (int, float)):
                    self.writer.add_scalar(
                        f"{attack_name}/{metric_name}",
                        metric_value,
                        global_step=0
                    )
        
        # Log model graph
        sample_input = torch.randn(1, 3, 512, 512)
        self.writer.add_graph(self.model, sample_input)
        
        # Log sample images
        if hasattr(self, 'sample_images'):
            self.writer.add_images(
                "sample_images",
                self.sample_images,
                global_step=0
            )
        
        self.writer.close()
        return results

# Use TensorBoard pipeline
pipeline = TensorboardPipeline(
    model=model,
    dataset=dataset,
    attacks=attacks,
    metrics=metrics,
    log_dir="./runs/experiment_001"
)

results = pipeline.run()
```

### MLflow Integration

```python
import mlflow
import mlflow.pytorch

class MLflowPipeline(SegmentationRobustnessPipeline):
    """Pipeline with MLflow integration."""
    
    def __init__(self, *args, experiment_name="segmentation-robustness", **kwargs):
        super().__init__(*args, **kwargs)
        
        # Set experiment
        mlflow.set_experiment(experiment_name)
        
        # Start run
        mlflow.start_run()
        
        # Log parameters
        mlflow.log_params({
            "model_type": type(self.model).__name__,
            "dataset_type": type(self.dataset).__name__,
            "batch_size": self.batch_size,
            "device": self.device
        })
    
    def run(self, save=True, show=False):
        """Run evaluation with MLflow logging."""
        results = super().run(save=save, show=show)
        
        # Log metrics
        for attack_name, attack_results in results.items():
            for metric_name, metric_value in attack_results.items():
                if isinstance(metric_value, (int, float)):
                    mlflow.log_metric(
                        f"{attack_name}_{metric_name}",
                        metric_value
                    )
        
        # Log model
        mlflow.pytorch.log_model(self.model, "model")
        
        # Log artifacts
        if save:
            mlflow.log_artifacts(self.output_dir)
        
        mlflow.end_run()
        return results

# Use MLflow pipeline
pipeline = MLflowPipeline(
    model=model,
    dataset=dataset,
    attacks=attacks,
    metrics=metrics,
    experiment_name="my-experiment"
)

results = pipeline.run()
```

## 🐛 Debugging and Profiling

### Performance Profiling

```python
import cProfile
import pstats
import io
from torch.profiler import profile, record_function, ProfilerActivity

class ProfiledPipeline(SegmentationRobustnessPipeline):
    """Pipeline with performance profiling."""
    
    def __init__(self, *args, profile_enabled=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.profile_enabled = profile_enabled
    
    def run(self, save=True, show=False):
        """Run evaluation with profiling."""
        if not self.profile_enabled:
            return super().run(save=save, show=show)
        
        # CPU profiling
        pr = cProfile.Profile()
        pr.enable()
        
        # GPU profiling
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=True
        ) as prof:
            with record_function("model_inference"):
                results = super().run(save=save, show=show)
        
        # Stop CPU profiling
        pr.disable()
        
        # Print CPU profiling results
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        ps.print_stats(20)
        print("CPU Profiling Results:")
        print(s.getvalue())
        
        # Print GPU profiling results
        print("GPU Profiling Results:")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        
        # Save profiling results
        prof.export_chrome_trace("trace.json")
        
        return results

# Use profiled pipeline
pipeline = ProfiledPipeline(
    model=model,
    dataset=dataset,
    attacks=attacks,
    metrics=metrics,
    profile_enabled=True
)

results = pipeline.run()
```

### Memory Profiling

```python
import tracemalloc
import psutil
import GPUtil

class MemoryProfiledPipeline(SegmentationRobustnessPipeline):
    """Pipeline with memory profiling."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.memory_stats = []
    
    def _log_memory_usage(self, stage: str):
        """Log current memory usage."""
        # CPU memory
        cpu_memory = psutil.virtual_memory()
        
        # GPU memory
        gpu_memory = None
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_stats()
        
        self.memory_stats.append({
            'stage': stage,
            'cpu_percent': cpu_memory.percent,
            'cpu_used_gb': cpu_memory.used / (1024**3),
            'gpu_allocated_gb': gpu_memory['allocated_bytes.all.current'] / (1024**3) if gpu_memory else 0,
            'gpu_reserved_gb': gpu_memory['reserved_bytes.all.current'] / (1024**3) if gpu_memory else 0
        })
    
    def run(self, save=True, show=False):
        """Run evaluation with memory profiling."""
        # Start memory tracking
        tracemalloc.start()
        
        # Log initial memory
        self._log_memory_usage("start")
        
        # Run evaluation
        results = super().run(save=save, show=show)
        
        # Log final memory
        self._log_memory_usage("end")
        
        # Get memory snapshot
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Print memory statistics
        print(f"Current memory usage: {current / 1024 / 1024:.1f} MB")
        print(f"Peak memory usage: {peak / 1024 / 1024:.1f} MB")
        
        # Print detailed memory stats
        print("\nMemory Usage by Stage:")
        for stat in self.memory_stats:
            print(f"  {stat['stage']}:")
            print(f"    CPU: {stat['cpu_percent']:.1f}% ({stat['cpu_used_gb']:.2f} GB)")
            print(f"    GPU: {stat['gpu_allocated_gb']:.2f} GB allocated, {stat['gpu_reserved_gb']:.2f} GB reserved")
        
        return results

# Use memory profiled pipeline
pipeline = MemoryProfiledPipeline(
    model=model,
    dataset=dataset,
    attacks=attacks,
    metrics=metrics
)

results = pipeline.run()
```

## 🚀 Production Deployment

### Docker Containerization

```dockerfile
# Dockerfile for production deployment
FROM pytorch/pytorch:2.0.0-cuda11.8-cudnn8-devel

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directory
RUN mkdir -p /app/data

# Expose port for API
EXPOSE 8000

# Set environment variables
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0

# Run application
CMD ["python", "app.py"]
```

### FastAPI Web Service

```python
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import torch
import io
from PIL import Image
import numpy as np

app = FastAPI(title="Segmentation Robustness API")

# Load model
model = load_model()
pipeline = SegmentationRobustnessPipeline(
    model=model,
    dataset=None,  # Will be created from uploaded data
    attacks=[],
    metrics=[]
)

@app.post("/evaluate")
async def evaluate_segmentation(
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
    attack_type: str = "fgsm",
    eps: float = 0.008
):
    """Evaluate segmentation robustness."""
    try:
        # Load image and mask
        image_data = await image.read()
        mask_data = await mask.read()
        
        # Convert to tensors
        image_tensor = load_image_tensor(image_data)
        mask_tensor = load_mask_tensor(mask_data)
        
        # Create attack
        attack = create_attack(attack_type, model, eps)
        
        # Run evaluation
        results = pipeline.evaluate_single_sample(
            image_tensor, mask_tensor, attack
        )
        
        return JSONResponse(content=results)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": model is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Kubernetes Deployment

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: segmentation-robustness
spec:
  replicas: 3
  selector:
    matchLabels:
      app: segmentation-robustness
  template:
    metadata:
      labels:
        app: segmentation-robustness
    spec:
      containers:
      - name: app
        image: segmentation-robustness:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
            nvidia.com/gpu: 1
          limits:
            memory: "8Gi"
            cpu: "4"
            nvidia.com/gpu: 1
        env:
        - name: CUDA_VISIBLE_DEVICES
          value: "0"
        volumeMounts:
        - name: data-volume
          mountPath: /app/data
      volumes:
      - name: data-volume
        persistentVolumeClaim:
          claimName: data-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: segmentation-robustness-service
spec:
  selector:
    app: segmentation-robustness
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

---

**This advanced usage guide covers high-performance and production-ready features. For more information:**

- 📖 [User Guide](user_guide.md) - General framework usage
- 🔧 [Custom Components](custom_components.md) - Creating custom components
- ⚙️ [API Reference](api_reference.md) - Complete API documentation 