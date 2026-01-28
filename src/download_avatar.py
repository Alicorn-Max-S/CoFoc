#!/usr/bin/env python3
"""
Avatar Download Utility for CoFoc

Downloads high-quality 3D avatar models from various sources:
- Ready Player Me (custom avatars)
- VRoid Hub (VRM VTuber models)
- Direct URLs (GLB/glTF/VRM files)

Usage:
    python download_avatar.py --source readyplayer --id YOUR_AVATAR_ID
    python download_avatar.py --source vroid --url VROID_HUB_URL
    python download_avatar.py --url https://example.com/model.glb
"""

import os
import sys
import argparse
import requests
from pathlib import Path
from urllib.parse import urlparse, parse_qs


MODELS_DIR = Path(__file__).parent.parent / "models"
DEFAULT_OUTPUT = MODELS_DIR / "avatar.glb"


def ensure_models_dir():
    """Create models directory if it doesn't exist."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def download_ready_player_me(avatar_id: str, output_path: Path, quality: str = "high") -> bool:
    """
    Download an avatar from Ready Player Me.

    Get your avatar ID by:
    1. Go to https://readyplayer.me/avatar
    2. Create or select an avatar
    3. Copy the avatar URL - the ID is the part before .glb

    Example: https://models.readyplayer.me/6185a4acfb622cf1cdc49348.glb
    The avatar ID is: 6185a4acfb622cf1cdc49348
    """
    print(f"Downloading Ready Player Me avatar: {avatar_id}")

    quality_map = {"high": 0, "medium": 1, "low": 2}
    mesh_lod = quality_map.get(quality, 0)

    url = f"https://models.readyplayer.me/{avatar_id}.glb"
    params = {
        "meshLod": mesh_lod,
        "pose": "A",  # A-pose for animation
        "morphTargets": "ARKit,Oculus Visemes",  # Include blend shapes for expressions
        "textureAtlas": "1024",  # Texture resolution
    }

    try:
        print(f"Fetching from: {url}")
        response = requests.get(url, params=params, timeout=60)
        response.raise_for_status()

        with open(output_path, 'wb') as f:
            f.write(response.content)

        size_mb = len(response.content) / (1024 * 1024)
        print(f"Downloaded {size_mb:.2f} MB to {output_path}")
        return True

    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: {e}")
        print("Make sure the avatar ID is correct.")
        print("You can get your avatar ID from: https://readyplayer.me/avatar")
        return False
    except Exception as e:
        print(f"Download failed: {e}")
        return False


def download_from_url(url: str, output_path: Path) -> bool:
    """Download a model from a direct URL."""
    print(f"Downloading from: {url}")

    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) CoFoc Avatar Downloader'
        }
        response = requests.get(url, headers=headers, timeout=60, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0

        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size:
                        percent = (downloaded / total_size) * 100
                        print(f"\rDownloading: {percent:.1f}%", end='', flush=True)

        print(f"\nSaved to: {output_path}")
        return True

    except Exception as e:
        print(f"Download failed: {e}")
        return False


def download_vroid_sample() -> bool:
    """
    Download official VRoid Studio CC0 sample models.

    These are free-to-use models released by Pixiv under CC0 license.
    """
    print("VRoid Studio Sample Models (CC0 License)")
    print("=" * 50)
    print("""
These official sample models are available from VRoid:

1. AvatarSample_A, B, C - Modern VRM 1.0 models
2. Beta Ver AvatarSample 1-4 - Classic VRM 0.x models

To use a VRoid model:

Option A - VRoid Hub:
  1. Visit https://hub.vroid.com
  2. Find a model you like (check the license!)
  3. Download the VRM file
  4. Place it in the 'models' folder as 'avatar.vrm'

Option B - Create Your Own:
  1. Download VRoid Studio (free): https://vroid.com/en/studio
  2. Create your custom avatar
  3. Export as VRM
  4. Place in 'models' folder

Option C - CC0 Sample Models:
  Visit: https://vroid.pixiv.help/hc/en-us/articles/4402394424089
  Download the sample models and place in 'models' folder
""")
    return True


def print_sources_info():
    """Print information about avatar sources."""
    print("""
=== AVATAR SOURCES ===

1. READY PLAYER ME (Recommended for beginners)
   - Website: https://readyplayer.me/avatar
   - Create a free customizable avatar
   - Supports full body, half body, or head only
   - High quality with blend shapes for expressions

   Usage:
     python download_avatar.py --source readyplayer --id YOUR_AVATAR_ID

2. VROID HUB (Best for anime/VTuber style)
   - Website: https://hub.vroid.com
   - Thousands of free VRM models
   - Many allow commercial use
   - VTuber-ready with expressions

   Usage:
     Download VRM file manually, then:
     python download_avatar.py --url /path/to/model.vrm

3. MIXAMO (Best for realistic humans)
   - Website: https://www.mixamo.com
   - Free rigged characters
   - Huge animation library
   - Export as FBX, convert to glTF with Blender

4. SKETCHFAB (Variety of styles)
   - Website: https://sketchfab.com
   - Search for "humanoid rigged" or "avatar"
   - Many free downloadable models
   - Check license before use

5. TURBOSQUID (Professional quality)
   - Website: https://www.turbosquid.com
   - Search for "free humanoid rigged glTF"
   - Professional quality models

=== SUPPORTED FORMATS ===
- GLB (recommended) - Binary glTF
- glTF - JSON + separate binary
- VRM - VTuber standard (based on glTF)

=== REQUIREMENTS FOR BEST RESULTS ===
- Full body humanoid model
- Properly rigged skeleton
- Standard bone names (Mixamo-compatible)
- Face blend shapes for expressions (optional)
""")


def main():
    parser = argparse.ArgumentParser(
        description="Download high-quality 3D avatar models for CoFoc",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download from Ready Player Me
  python download_avatar.py --source readyplayer --id 6185a4acfb622cf1cdc49348

  # Download from a direct URL
  python download_avatar.py --url https://example.com/avatar.glb

  # Show information about avatar sources
  python download_avatar.py --info
        """
    )

    parser.add_argument('--source', choices=['readyplayer', 'vroid', 'url'],
                        help='Avatar source')
    parser.add_argument('--id', help='Avatar ID (for Ready Player Me)')
    parser.add_argument('--url', help='Direct URL to download')
    parser.add_argument('--output', '-o', type=Path, default=DEFAULT_OUTPUT,
                        help=f'Output path (default: {DEFAULT_OUTPUT})')
    parser.add_argument('--quality', choices=['high', 'medium', 'low'], default='high',
                        help='Quality level for Ready Player Me (default: high)')
    parser.add_argument('--info', action='store_true',
                        help='Show information about avatar sources')

    args = parser.parse_args()

    if args.info:
        print_sources_info()
        return

    ensure_models_dir()

    if args.source == 'readyplayer':
        if not args.id:
            print("Error: --id is required for Ready Player Me")
            print("Get your avatar ID from: https://readyplayer.me/avatar")
            sys.exit(1)
        success = download_ready_player_me(args.id, args.output, args.quality)

    elif args.source == 'vroid':
        success = download_vroid_sample()

    elif args.url:
        output = args.output
        # Detect file extension from URL if not specified
        if output == DEFAULT_OUTPUT:
            parsed = urlparse(args.url)
            ext = Path(parsed.path).suffix.lower()
            if ext in ['.glb', '.gltf', '.vrm']:
                output = MODELS_DIR / f"avatar{ext}"
        success = download_from_url(args.url, output)

    else:
        print("No source specified. Use --info to see available options.")
        parser.print_help()
        sys.exit(1)

    if success:
        print("\nAvatar downloaded successfully!")
        print(f"Run 'python src/main.py' to see your avatar.")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
