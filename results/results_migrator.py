import os
import json
import shutil
from pathlib import Path
from datetime import datetime, timezone
import re
import argparse
from typing import Dict, List, Optional, Tuple
import git
from collections import defaultdict

class VersionBasedMigrator:
    def __init__(self, source_dir: str, repo_path: str = None, backup: bool = True):
        self.source_dir = Path(source_dir)
        self.repo_path = repo_path or self._find_git_repo()
        self.backup = backup
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.repo = git.Repo(self.repo_path)
        self.commit_map = self._build_commit_map()
        
    def _find_git_repo(self) -> str:
        """Find the git repository by walking up the directory tree."""
        current_dir = self.source_dir
        while current_dir != current_dir.parent:
            if (current_dir / '.git').exists():
                return str(current_dir)
            current_dir = current_dir.parent
        raise ValueError("Could not find git repository")

    def _build_commit_map(self) -> Dict[str, List[git.Commit]]:
        """Build a mapping of relevant files to their commit history."""
        relevant_files = [
            'classifier.py',
            'UMAP.py',
            'distributions.py',
            'get_embeddings.py'
        ]
        
        commit_map = defaultdict(list)
        
        # Get commit history for each relevant file
        for file in relevant_files:
            try:
                # Get the full history of the file
                for commit in self.repo.iter_commits(paths=file):
                    commit_time = datetime.fromtimestamp(commit.committed_date)
                    commit_map[file].append({
                        'hash': commit.hexsha,
                        'timestamp': commit_time,
                        'message': commit.message.strip(),
                        'file': file
                    })
            except git.exc.GitCommandError:
                print(f"Warning: Could not get history for {file}")
                
        return commit_map

    def _find_active_commit(self, timestamp: datetime) -> Dict:
        """Find which commit was active at a given timestamp."""
        best_matches = []
        
        # Check each tracked file
        for file, commits in self.commit_map.items():
            # Find the most recent commit before the timestamp
            active_commit = None
            for commit in commits:
                if commit['timestamp'] <= timestamp:
                    active_commit = commit
                    break
            
            if active_commit:
                best_matches.append(active_commit)
        
        # If we found matches, return the most recent one
        if best_matches:
            return sorted(best_matches, key=lambda x: x['timestamp'], reverse=True)[0]
        
        return None

    def _parse_filename(self, filename: str) -> Tuple[str, datetime, str]:
        """Parse a result filename into its components."""
        pattern = r"([A-Za-z]+)_(\d{8})_(\d{6})_([a-f0-9]+)"
        match = re.match(pattern, filename)
        
        if match:
            result_type, date, time, config_hash = match.groups()
            timestamp = datetime.strptime(f"{date}_{time}", "%Y%m%d_%H%M%S")
            return result_type.lower(), timestamp, config_hash
        return None, None, None

    def _create_version_summary(self, version_dir: Path, commit_info: Dict) -> Dict:
        """Create a summary of what's in each version directory."""
        return {
            "commit_hash": commit_info['hash'][:8],
            "commit_date": commit_info['timestamp'].isoformat(),
            "commit_message": commit_info['message'],
            "modified_file": commit_info['file'],
            "results_count": {
                "classifier": len(list(version_dir.glob("classifier/*"))),
                "umap": len(list(version_dir.glob("UMAP/*"))),
                "tsne": len(list(version_dir.glob("tsne/*"))),
                "distributions": len(list(version_dir.glob("distributions/*")))
            }
        }

    def migrate(self):
        """Migrate existing results to version-based structure."""
        print("Starting version-based migration...")
        
        if self.backup:
            backup_dir = self.source_dir.parent / f"results_backup_{self.timestamp}"
            shutil.copytree(self.source_dir, backup_dir)
            print(f"Created backup at: {backup_dir}")

        # Create new structure
        versions_dir = self.source_dir / "versions"
        versions_dir.mkdir(exist_ok=True)
        
        # Track which results go with which versions
        version_mapping = defaultdict(list)
        
        # First pass: analyze files and determine their versions
        for file_path in self.source_dir.rglob("*"):
            if not file_path.is_file() or file_path.name.startswith('.'):
                continue
                
            result_type, timestamp, config_hash = self._parse_filename(file_path.name)
            if not all([result_type, timestamp, config_hash]):
                continue
                
            # Find the active commit for this result
            commit_info = self._find_active_commit(timestamp)
            if commit_info:
                version_id = f"v_{commit_info['hash'][:8]}"
                version_mapping[version_id].append({
                    'file': file_path,
                    'type': result_type,
                    'config_hash': config_hash,
                    'commit_info': commit_info
                })

        # Second pass: organize files by version
        version_summaries = {}
        for version_id, files in version_mapping.items():
            if not files:
                continue
                
            version_dir = versions_dir / version_id
            version_dir.mkdir(exist_ok=True)
            
            # Create type directories under version
            for result_type in ['classifier', 'umap', 'tsne', 'distributions']:
                (version_dir / result_type).mkdir(exist_ok=True)
            
            # Move files
            for file_info in files:
                file_path = file_info['file']
                result_type = file_info['type']
                target_dir = version_dir / result_type
                
                # Create result-specific directory if needed
                result_dir = target_dir / file_path.stem
                result_dir.mkdir(exist_ok=True)
                
                # Move the file
                new_path = result_dir / file_path.name
                shutil.move(str(file_path), str(new_path))
            
            # Create version summary
            version_summaries[version_id] = self._create_version_summary(
                version_dir, files[0]['commit_info']
            )

        # Save version index
        index = {
            "migration_date": self.timestamp,
            "versions": version_summaries
        }
        
        with open(self.source_dir / "version_index.json", 'w') as f:
            json.dump(index, f, indent=2)

        print("\nMigration complete!")
        print(f"- Organized results into {len(version_summaries)} versions")
        print(f"- Version index saved to: {self.source_dir/'version_index.json'}")
        if self.backup:
            print(f"- Backup created with timestamp: {self.timestamp}")

def main():
    parser = argparse.ArgumentParser(description="Migrate results to version-based structure")
    parser.add_argument("--source", default="results",
                      help="Source directory containing results")
    parser.add_argument("--repo", default=None,
                      help="Path to git repository (default: auto-detect)")
    parser.add_argument("--no-backup", action="store_true",
                      help="Skip creating backup before migration")
    
    args = parser.parse_args()
    
    migrator = VersionBasedMigrator(
        args.source, 
        repo_path=args.repo, 
        backup=not args.no_backup
    )
    migrator.migrate()

if __name__ == "__main__":
    main()