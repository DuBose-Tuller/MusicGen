import os
import json
import shutil
from pathlib import Path
from datetime import datetime
import git
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import yaml
from tabulate import tabulate

@dataclass
class ResultMetrics:
    """Container for result metrics across versions."""
    version: str
    type: str
    metrics: Dict
    config: Dict
    timestamp: datetime

class VersionResultsManager:
    def __init__(self, base_dir: str, repo_path: str = None):
        self.base_dir = Path(base_dir)
        self.versions_dir = self.base_dir / "versions"
        self.repo_path = repo_path or self._find_git_repo()
        self.repo = git.Repo(self.repo_path)
        
    def add_result(self, 
                  files: Dict[str, Path],
                  metadata: Dict,
                  result_type: str,
                  tags: Optional[List[str]] = None) -> str:
        """Add a new result to the current version."""
        current_commit = self.repo.head.commit
        version_id = f"v_{current_commit.hexsha[:8]}"
        
        # Create version directory if it doesn't exist
        version_dir = self.versions_dir / version_id
        version_dir.mkdir(parents=True, exist_ok=True)
        
        # Create result directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_id = f"{result_type}_{timestamp}_{current_commit.hexsha[:8]}"
        result_dir = version_dir / result_type / result_id
        result_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy files to result directory
        for file_type, file_path in files.items():
            shutil.copy2(file_path, result_dir / file_path.name)
            
        # Save metadata
        metadata.update({
            "commit_hash": current_commit.hexsha,
            "commit_message": current_commit.message,
            "timestamp": timestamp,
            "tags": tags or []
        })
        
        with open(result_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
            
        return result_id

    def compare_versions(self, 
                        versions: List[str], 
                        result_type: str,
                        metric_keys: Optional[List[str]] = None) -> pd.DataFrame:
        """Compare metrics across versions for a specific result type."""
        metrics_data = []
        
        for version in versions:
            version_dir = self.versions_dir / version
            if not version_dir.exists():
                continue
                
            results_dir = version_dir / result_type
            if not results_dir.exists():
                continue
                
            # Collect metrics from all results in this version
            for result_dir in results_dir.iterdir():
                metadata_file = result_dir / "metadata.json"
                if not metadata_file.exists():
                    continue
                    
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    
                metrics = metadata.get("metrics", {})
                if metric_keys:
                    metrics = {k: metrics.get(k) for k in metric_keys}
                    
                metrics_data.append({
                    "version": version,
                    "result_id": result_dir.name,
                    "timestamp": metadata["timestamp"],
                    **metrics
                })
                
        return pd.DataFrame(metrics_data)

    def generate_report(self, 
                       output_file: str = "version_report.md",
                       versions: Optional[List[str]] = None):
        """Generate a comprehensive markdown report of results across versions."""
        if versions is None:
            versions = sorted([d.name for d in self.versions_dir.iterdir() if d.is_dir()])
            
        report_parts = ["# Results Analysis Report\n"]
        
        # Version Overview
        report_parts.append("## Version Overview\n")
        for version in versions:
            commit = self.repo.commit(version[2:])  # Remove 'v_' prefix
            report_parts.extend([
                f"### {version}",
                f"- Commit: {commit.hexsha[:8]}",
                f"- Date: {datetime.fromtimestamp(commit.committed_date)}",
                f"- Message: {commit.message.strip()}",
                ""
            ])
            
        # Results Summary
        report_parts.append("## Results Summary\n")
        for result_type in ['classifier', 'umap', 'tsne', 'distributions']:
            report_parts.append(f"### {result_type.upper()} Results\n")
            
            metrics_df = self.compare_versions(versions, result_type)
            if not metrics_df.empty:
                summary = metrics_df.groupby('version').agg({
                    col: ['mean', 'std'] for col in metrics_df.columns 
                    if col not in ['version', 'result_id', 'timestamp']
                })
                report_parts.append("```\n" + tabulate(summary, headers='keys', tablefmt='pipe') + "\n```\n")
            
        # Write report
        with open(output_file, 'w') as f:
            f.write('\n'.join(report_parts))

    def visualize_trends(self,
                        metric: str,
                        result_type: str,
                        versions: Optional[List[str]] = None,
                        output_file: Optional[str] = None):
        """Visualize trends in a specific metric across versions."""
        if versions is None:
            versions = sorted([d.name for d in self.versions_dir.iterdir() if d.is_dir()])
            
        metrics_df = self.compare_versions(versions, result_type)
        if metric not in metrics_df.columns:
            raise ValueError(f"Metric '{metric}' not found in results")
            
        plt.figure(figsize=(12, 6))
        
        # Create box plot
        sns.boxplot(data=metrics_df, x='version', y=metric)
        plt.xticks(rotation=45)
        plt.title(f"{metric} Across Versions")
        
        if output_file:
            plt.savefig(output_file, bbox_inches='tight')
        plt.close()

    def analyze_changes(self, 
                       version1: str, 
                       version2: str, 
                       result_type: str) -> Dict:
        """Analyze changes in results between two versions."""
        metrics_v1 = self.compare_versions([version1], result_type)
        metrics_v2 = self.compare_versions([version2], result_type)
        
        changes = {}
        
        # Find common metrics
        common_metrics = [col for col in metrics_v1.columns 
                        if col in metrics_v2.columns 
                        and col not in ['version', 'result_id', 'timestamp']]
        
        for metric in common_metrics:
            v1_values = metrics_v1[metric].dropna()
            v2_values = metrics_v2[metric].dropna()
            
            if len(v1_values) == 0 or len(v2_values) == 0:
                continue
                
            # Calculate statistics
            changes[metric] = {
                'mean_change': float(v2_values.mean() - v1_values.mean()),
                'mean_change_percent': float((v2_values.mean() - v1_values.mean()) / v1_values.mean() * 100),
                'std_change': float(v2_values.std() - v1_values.std()),
                'statistical_test': {
                    'test': 'mann_whitney',
                    'p_value': float(scipy.stats.mannwhitneyu(v1_values, v2_values).pvalue)
                }
            }
            
        return changes

def main():
    parser = argparse.ArgumentParser(description="Version-based results management")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Add result command
    add_parser = subparsers.add_parser("add", help="Add new result")
    add_parser.add_argument("--type", required=True, help="Type of result")
    add_parser.add_argument("--files", required=True, nargs="+", help="Files to add")
    add_parser.add_argument("--metadata", required=True, help="Path to metadata file")
    add_parser.add_argument("--tags", nargs="+", help="Optional tags")
    
    # Compare versions command
    compare_parser = subparsers.add_parser("compare", help="Compare versions")
    compare_parser.add_argument("--versions", nargs="+", required=True, help="Versions to compare")
    compare_parser.add_argument("--type", required=True, help="Type of result")
    compare_parser.add_argument("--metric", required=True, help="Metric to compare")
    compare_parser.add_argument("--output", help="Output file for visualization")
    
    # Generate report command
    report_parser = subparsers.add_parser("report", help="Generate analysis report")
    report_parser.add_argument("--output", default="version_report.md", help="Output file")
    report_parser.add_argument("--versions", nargs="+", help="Specific versions to include")
    
    args = parser.parse_args()
    manager = VersionResultsManager("results")
    
    if args.command == "add":
        files = {Path(f).suffix[1:]: Path(f) for f in args.files}
        with open(args.metadata, 'r') as f:
            metadata = json.load(f)
        result_id = manager.add_result(files, metadata, args.type, args.tags)
        print(f"Added result: {result_id}")
        
    elif args.command == "compare":
        manager.visualize_trends(args.metric, args.type, args.versions, args.output)
        print(f"Visualization saved to {args.output}")
        
    elif args.command == "report":
        manager.generate_report(args.output, args.versions)
        print(f"Report generated: {args.output}")

if __name__ == "__main__":
    main()