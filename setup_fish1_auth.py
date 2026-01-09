#!/usr/bin/env python3
"""
Setup script for Fish1 (Zebrafish) dataset authentication.

This script helps you configure authentication for accessing the Fish1 zebrafish dataset.

Steps:
1. Request access: https://forms.gle/tpbndoL1J6xB47KQ9 (if not already done)
2. Get your token from ONE of these URLs (after approval):
   - New token: https://global.brain-wire-test.org/auth/api/v1/create_token
   - Existing token: https://global.brain-wire-test.org/auth/api/v1/user/token
3. Run this script and paste your token when prompted
"""

import caveclient
import sys


def setup_cave_auth():
    """Setup CAVE authentication for Fish1 and H01 datasets."""

    print("=" * 70)
    print("CAVE Authentication Setup for Fish1 (Zebrafish) Dataset")
    print("=" * 70)
    print()

    # Check if user has requested access
    print("Step 1: Access Request")
    print("-" * 70)
    print("Have you requested access to the Fish1 dataset?")
    print("If not, please submit this form: https://forms.gle/tpbndoL1J6xB47KQ9")
    print("You will receive a response within 24 hours.")
    print()

    response = input("Have you been approved for access? (yes/no): ").strip().lower()
    if response not in ['yes', 'y']:
        print("\nPlease request access first, then run this script again.")
        sys.exit(0)

    print()
    print("Step 2: Get Your Token")
    print("-" * 70)
    print("Please visit ONE of the following URLs to get your token:")
    print()
    print("  For NEW users (first time):")
    print("    https://global.brain-wire-test.org/auth/api/v1/create_token")
    print()
    print("  For EXISTING users (already have a token):")
    print("    https://global.brain-wire-test.org/auth/api/v1/user/token")
    print()
    print("Log in with your Google credentials and copy the token.")
    print("The token looks like: 4836d842ee67473399ca468f61d773ff")
    print()

    token = input("Paste your token here: ").strip()

    if not token:
        print("\nError: No token provided. Exiting.")
        sys.exit(1)

    print()
    print("Step 3: Save Token")
    print("-" * 70)

    # Initialize auth client
    url = "https://global.brain-wire-test.org/"
    auth = caveclient.auth.AuthClient(server_address=url)

    # Save the token
    try:
        auth.save_token(token=token, overwrite=True)
        print("✓ Token saved successfully!")
    except Exception as e:
        print(f"✗ Error saving token: {e}")
        sys.exit(1)

    print()
    print("Step 4: Testing Connection")
    print("-" * 70)

    # Test H01 connection
    try:
        print("Testing H01 (Human) dataset...")
        h01_client = caveclient.CAVEclient(datastack_name='h01_c3_flat', server_address=url)
        print(f"  ✓ Successfully connected to H01: {h01_client.datastack_name}")
    except Exception as e:
        print(f"  ✗ Failed to connect to H01: {e}")

    print()

    # Test Fish1 connection
    try:
        print("Testing Fish1 (Zebrafish) dataset...")
        fish1_client = caveclient.CAVEclient(datastack_name='fish1_full', server_address=url)
        print(f"  ✓ Successfully connected to Fish1: {fish1_client.datastack_name}")

        # Test that we can actually query the info service
        print("\nTesting data access...")
        em_path = fish1_client.info.image_source()
        seg_path = fish1_client.info.segmentation_source()
        print(f"  ✓ EM path: {em_path}")
        print(f"  ✓ Segmentation path: {seg_path}")

    except Exception as e:
        print(f"  ✗ Failed to connect to Fish1: {e}")
        sys.exit(1)

    print()
    print("=" * 70)
    print("SUCCESS! Authentication is configured correctly.")
    print("=" * 70)
    print()
    print("You can now use the ConnectomeVisualizer with species='zebrafish'")
    print()
    print("Example:")
    print("  from src.connectome_visualizer import ConnectomeVisualizer")
    print("  visualizer = ConnectomeVisualizer(species='zebrafish')")
    print("  visualizer.load_neurons([864691128652664051])")
    print()


if __name__ == "__main__":
    setup_cave_auth()
