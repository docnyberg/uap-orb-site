#!/usr/bin/env python3
"""
create_thumb_videos.py - Create MP4 videos from thumbnail sequences
Standalone version that works with existing atlas directories
"""

import argparse
import subprocess
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re
import sys
import csv

# Import tkinter for GUI
import tkinter as tk
from tkinter import filedialog, messagebox

def natural_sort_key(text: str) -> List:
    """Generate key for natural sorting (handles numbers in filenames properly)"""
    def convert(t: str):
        return int(t) if t.isdigit() else t.lower()
    return [convert(c) for c in re.split(r'(\d+)', text)]

def find_ffmpeg() -> Optional[str]:
    """Find ffmpeg in PATH or common locations"""
    ffmpeg = shutil.which('ffmpeg')
    if ffmpeg:
        return ffmpeg
    
    # Windows common locations
    common_paths = [
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\Tools\ffmpeg\bin\ffmpeg.exe",
    ]
    
    for path in common_paths:
        if Path(path).exists():
            return path
    
    return None

def create_video_from_sequence(
    frame_dir: Path,
    pattern: str,
    output_path: Path,
    fps: int = 30,
    ffmpeg_path: Optional[str] = None
) -> Tuple[bool, str]:
    """
    Create MP4 video from image sequence
    Returns (success, message)
    """
    if not ffmpeg_path:
        ffmpeg_path = find_ffmpeg()
    
    if not ffmpeg_path:
        return False, "ffmpeg not found. Please install ffmpeg and add it to PATH"
    
    # Get list of frames
    frames = sorted(frame_dir.glob(pattern), key=lambda x: natural_sort_key(x.name))
    
    if not frames:
        return False, f"No frames found matching pattern: {pattern}"
    
    # Create a file list for ffmpeg
    list_file = output_path.parent / f"{output_path.stem}_files.txt"
    with open(list_file, 'w') as f:
        for frame in frames:
            # FFmpeg concat format
            f.write(f"file '{frame.absolute()}'\n")
            f.write(f"duration 0.033\n")  # ~30fps
    
    cmd = [
        ffmpeg_path,
        '-f', 'concat',
        '-safe', '0',
        '-i', str(list_file),
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-preset', 'medium',
        '-crf', '23',
        '-movflags', '+faststart',
        '-y',  # Overwrite output
        str(output_path)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        list_file.unlink()  # Clean up temp file
        
        if result.returncode != 0:
            return False, f"ffmpeg error: {result.stderr}"
        return True, "Success"
    except Exception as e:
        if list_file.exists():
            list_file.unlink()
        return False, f"Error running ffmpeg: {str(e)}"

def analyze_atlas_directory(atlas_dir: Path, thumbs_subdir: str = 'thumbs_obj') -> Dict[str, List[Path]]:
    """
    Analyze atlas directory to find frame sequences
    Returns dict of {object_id: [frame_paths]}
    """
    sequences = {}
    thumbs_dir = atlas_dir / thumbs_subdir  # Changed to use parameter
    
    if not thumbs_dir.exists():
        print(f"Warning: {thumbs_dir} does not exist")
        return sequences
    
    # Find all frame files
    frame_files = list(thumbs_dir.glob('*_frame_*.jpg'))
    
    # Group by object ID
    for frame_file in frame_files:
        # Extract object ID by removing _frame_XXXXXX.jpg
        match = re.match(r'(.+)_frame_\d+\.jpg$', frame_file.name)
        if match:
            obj_id = match.group(1)
            if obj_id not in sequences:
                sequences[obj_id] = []
            sequences[obj_id].append(frame_file)
    
    # Sort frames in each sequence
    for obj_id in sequences:
        sequences[obj_id].sort(key=lambda x: natural_sort_key(x.name))
    
    return sequences

def process_atlas_directory(
    atlas_dir: Path,
    output_dir: Optional[Path] = None,
    fps: int = 30,
    create_web_versions: bool = True,
    thumbs_subdir: str = 'thumbs_obj'  # Changed default from 'thumbs' to 'thumbs_obj'
) -> Dict[str, List[str]]:
    """
    Process an atlas directory and create videos for each object
    """
    if not output_dir:
        output_dir = atlas_dir / 'videos'
    
    output_dir.mkdir(exist_ok=True)
    
    # Use specified subdirectory
    frames_dir = atlas_dir / thumbs_subdir
    
    # Analyze directory for sequences
    print(f"Analyzing {frames_dir} for frame sequences...")
    sequences = analyze_atlas_directory(atlas_dir, thumbs_subdir)
    
    if not sequences:
        print("No frame sequences found in atlas directory")
        print(f"Checked: {atlas_dir / thumbs_subdir}")
        return {'created': [], 'failed': [], 'skipped': []}
    
    print(f"Found {len(sequences)} objects with frame sequences")
    
    results = {
        'created': [],
        'failed': [],
        'skipped': []
    }
    
    ffmpeg_path = find_ffmpeg()
    if not ffmpeg_path:
        print("ERROR: ffmpeg not found. Please install ffmpeg.")
        return results
    
    # Process each sequence
    for obj_id, frames in sequences.items():
        print(f"\nProcessing {obj_id}...")
        print(f"  Found {len(frames)} frames")
        
        if len(frames) < 2:
            print(f"  Skipping - need at least 2 frames for video")
            results['skipped'].append(obj_id)
            continue
        
        # Create pattern for this object
        pattern = f"{obj_id}_frame_*.jpg"
        
        # Create video
        video_path = output_dir / f"{obj_id}.mp4"
        success, msg = create_video_from_sequence(
            atlas_dir / thumbs_subdir,  # Changed from hardcoded 'thumbs'
            pattern,
            video_path,
            fps,
            ffmpeg_path
        )
        
        if success:
            print(f"  ✓ Created {video_path.name}")
            results['created'].append(str(video_path))
            
            # Create web-compatible version if needed
            if create_web_versions:
                # For now, our main video is already web-compatible
                # Could add additional processing here if needed
                pass
        else:
            print(f"  ✗ Failed to create video: {msg}")
            results['failed'].append(f"{obj_id}: {msg}")
    
    return results

def gui_select_directory():
    """Show GUI to select atlas directory"""
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    
    # Show directory selection dialog
    atlas_dir = filedialog.askdirectory(
        title="Select Atlas Directory (contains thumbs/ with frame sequences)"
    )
    
    if not atlas_dir:
        messagebox.showinfo("Cancelled", "No directory selected. Exiting.")
        root.destroy()
        sys.exit(0)
    
    # Validate it's an atlas directory
    atlas_path = Path(atlas_dir)
    if not (atlas_path / 'thumbs_obj').exists():  # Changed from 'thumbs' to 'thumbs_obj'
        messagebox.showerror(
            "Invalid Directory", 
            "Selected directory doesn't appear to be an atlas directory.\n"
            "It should contain a 'thumbs_obj' subdirectory."
        )
        root.destroy()
        sys.exit(1)
    
    root.destroy()
    return atlas_path

def main():
    parser = argparse.ArgumentParser(
        description='Create MP4 videos from thumbnail frame sequences in atlas directory'
    )
    parser.add_argument('atlas_dir', nargs='?', help='Path to atlas directory')
    parser.add_argument('--output', '-o', help='Output directory for videos (default: atlas_dir/videos)')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second (default: 30)')
    parser.add_argument('--no-web-convert', action='store_true', 
                       help='Skip creating web-compatible versions')
    
    args = parser.parse_args()
    
    # If no atlas_dir provided, show GUI
    if not args.atlas_dir:
        atlas_dir = gui_select_directory()
    else:
        atlas_dir = Path(args.atlas_dir)
        if not atlas_dir.exists():
            print(f"Error: Directory {atlas_dir} does not exist")
            return 1
    
    output_dir = Path(args.output) if args.output else None
    
    try:
        results = process_atlas_directory(
            atlas_dir,
            output_dir,
            args.fps,
            not args.no_web_convert
        )
        
        print("\n" + "="*60)
        print("Summary:")
        print(f"  Created: {len(results['created'])} videos")
        print(f"  Failed: {len(results['failed'])}")
        print(f"  Skipped: {len(results['skipped'])}")
        
        if results['failed']:
            print("\nFailed videos:")
            for fail in results['failed']:
                print(f"  - {fail}")
        
        # Show completion message in GUI if we used the picker
        if not args.atlas_dir:
            root = tk.Tk()
            root.withdraw()
            messagebox.showinfo(
                "Complete",
                f"Video creation complete!\n\n"
                f"Created: {len(results['created'])} videos\n"
                f"Failed: {len(results['failed'])}\n"
                f"Skipped: {len(results['skipped'])}\n\n"
                f"Videos saved to: {atlas_dir / 'videos'}"
            )
            root.destroy()
        
        return 0 if not results['failed'] else 1
        
    except Exception as e:
        print(f"Error: {str(e)}")
        if not args.atlas_dir:  # Show error in GUI if we used picker
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror("Error", f"An error occurred:\n{str(e)}")
            root.destroy()
        return 1

if __name__ == '__main__':
    sys.exit(main())