#!/usr/bin/env python3
"""
Setup script for CAVE authentication (H01 Human & Fish1 Zebrafish datasets).

This script helps you configure authentication for accessing restricted datasets.

Usage:
    python scripts/setup_cave_auth.py

Requirements:
    1. You must have already requested access via: https://forms.gle/tpbndoL1J6xB47KQ9
    2. You must have received approval from the Lichtman Lab team
    3. You need to log in with the Gmail account that was approved
"""

import sys
import os
from pathlib import Path

try:
    import caveclient
except ImportError:
    print("Error: caveclient not installed. Please run: pip install caveclient")
    sys.exit(1)

try:
    from dotenv import load_dotenv, set_key
except ImportError:
    print("Error: python-dotenv not installed. Please run: pip install python-dotenv")
    sys.exit(1)


def check_access_request():
    """Check if user has requested access."""
    print("=" * 70)
    print("CAVE Authentication Setup")
    print("For H01 (Human) and Fish1 (Zebrafish) Datasets")
    print("=" * 70)
    print()
    print("STEP 1: Access Request")
    print("-" * 70)
    print("Have you already requested access and been approved?")
    print()
    print("If not, please:")
    print("  1. Submit this form: https://forms.gle/tpbndoL1J6xB47KQ9")
    print("  2. Wait for approval (usually within 24 hours)")
    print("  3. Return to run this script again")
    print()

    response = input("Have you been approved for access? (yes/no): ").strip().lower()
    if response not in ['yes', 'y']:
        print("\nPlease request access first, then run this script again.")
        print("Exiting...")
        sys.exit(0)


def get_token_from_user():
    """Guide user to get their token."""
    print()
    print("STEP 2: Get Your Token")
    print("-" * 70)
    print("You need to get a token from the CAVE authentication server.")
    print()
    print("Choose ONE of the following options:")
    print()
    print("  Option A - For NEW users (first time setup):")
    print("    Visit: https://global.brain-wire-test.org/auth/api/v1/create_token")
    print("    Note: This will invalidate any previous tokens!")
    print()
    print("  Option B - For EXISTING users (already have a token on another computer):")
    print("    Visit: https://global.brain-wire-test.org/auth/api/v1/user/token")
    print()
    print("After visiting the URL:")
    print("  1. Log in with your approved Gmail account")
    print("  2. Copy the token (looks like: 4836d842ee67473399ca468f61d773ff)")
    print()

    token = input("Paste your token here: ").strip()

    if not token:
        print("\nError: No token provided.")
        sys.exit(1)

    if len(token) < 20:
        print("\nWarning: Token seems too short. Make sure you copied the full token.")
        response = input("Continue anyway? (yes/no): ").strip().lower()
        if response not in ['yes', 'y']:
            sys.exit(1)

    return token


def save_token_to_env(token):
    """Save token to .env file."""
    print()
    print("STEP 3: Save Token to .env File")
    print("-" * 70)

    # Find project root (where .env should be)
    project_root = Path(__file__).parent.parent
    env_file = project_root / ".env"

    # Check if .env exists
    if env_file.exists():
        print(f"Found existing .env file at: {env_file}")
        response = input("Update existing .env file? (yes/no): ").strip().lower()
        if response not in ['yes', 'y']:
            print("\nCancelled. Token not saved.")
            sys.exit(0)
    else:
        print(f"Creating new .env file at: {env_file}")

    try:
        # Use set_key to add or update the token
        set_key(env_file, "CAVECLIENT_TOKEN", token)
        print(f"✓ Token saved successfully to: {env_file}")
    except Exception as e:
        print(f"✗ Error saving token to .env file: {e}")
        print("\nPlease manually add this line to your .env file:")
        print(f"  CAVECLIENT_TOKEN={token}")
        sys.exit(1)

    # Also save to CAVEclient's default location for compatibility
    try:
        url = "https://global.brain-wire-test.org/"
        auth = caveclient.auth.AuthClient(server_address=url)
        auth.save_token(token=token, overwrite=True)
        print("✓ Token also saved to CAVEclient default location")
    except Exception as e:
        print(f"Note: Could not save to CAVEclient default location: {e}")
        print("This is OK - the .env file will be used instead.")


def test_connection(token):
    """Test the connection to both H01 and Fish1 datasets."""
    print()
    print("STEP 4: Testing Connection")
    print("-" * 70)

    url = "https://global.brain-wire-test.org/"

    # Test H01 (Human)
    print("\nTesting H01 (Human) dataset...")
    try:
        h01_client = caveclient.CAVEclient(
            datastack_name='h01_c3_flat',
            server_address=url,
            auth_token=token
        )
        print(f"  ✓ Connected to H01: {h01_client.datastack_name}")

        # Try to query info service
        try:
            em_path = h01_client.info.image_source()
            seg_path = h01_client.info.segmentation_source()
            print(f"  ✓ InfoService accessible")
        except Exception as e:
            print(f"  ⚠ InfoService error: {e}")
    except Exception as e:
        print(f"  ✗ Failed to connect to H01: {e}")
        print("  This might indicate an issue with your token or permissions.")

    # Test Fish1 (Zebrafish)
    print("\nTesting Fish1 (Zebrafish) dataset...")
    try:
        fish1_client = caveclient.CAVEclient(
            datastack_name='fish1_full',
            server_address=url,
            auth_token=token
        )
        print(f"  ✓ Connected to Fish1: {fish1_client.datastack_name}")

        # Try to query info service
        try:
            em_path = fish1_client.info.image_source()
            seg_path = fish1_client.info.segmentation_source()
            print(f"  ✓ InfoService accessible")
            print(f"    EM: {em_path}")
            print(f"    Segmentation: {seg_path}")
        except Exception as e:
            print(f"  ⚠ InfoService error: {e}")
    except Exception as e:
        print(f"  ✗ Failed to connect to Fish1: {e}")
        print("  This might indicate an issue with your token or permissions.")


def print_success_message():
    """Print final success message with usage examples."""
    print()
    print("=" * 70)
    print("SUCCESS! Authentication is configured.")
    print("=" * 70)
    print()
    print("You can now use ConnectomeVisualizer with human and zebrafish datasets:")
    print()
    print("Example usage:")
    print("  from src.connectome_visualizer import ConnectomeVisualizer")
    print()
    print("  # Zebrafish")
    print("  visualizer = ConnectomeVisualizer(species='zebrafish')")
    print("  visualizer.load_neurons([864691128652664051])")
    print("  visualizer.save_3d_views(base_filename='zebrafish_neuron')")
    print()
    print("  # Human")
    print("  visualizer = ConnectomeVisualizer(species='human')")
    print("  visualizer.load_neurons([864691135274541057])")
    print("  visualizer.save_3d_views(base_filename='human_neuron')")
    print()
    print("Note: Make sure to run Python from the project root directory")
    print("so it can find the .env file.")
    print()


def main():
    """Main setup flow."""
    try:
        # Step 1: Check if user has requested access
        check_access_request()

        # Step 2: Get token from user
        token = get_token_from_user()

        # Step 3: Save token to .env
        save_token_to_env(token)

        # Step 4: Test connection
        test_connection(token)

        # Print success message
        print_success_message()

    except KeyboardInterrupt:
        print("\n\nSetup cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
