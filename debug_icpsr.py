#!/usr/bin/env python3
"""Debug ICPSR collection issues."""

from pathlib import Path


def debug_icpsr():
    """Debug ICPSR collection step by step."""
    try:
        from softverse.collectors.icpsr_collector import ICPSRScriptCollector
        from softverse.utils.file_utils import get_disk_usage
        from softverse.utils.logging_utils import ArchiveProgressLogger, get_logger

        print("Creating ICPSR collector...")
        collector = ICPSRScriptCollector()

        print("Searching for studies...")
        studies_df = collector.search_icpsr_aea(max_records=2)
        if studies_df.empty:
            print("No studies found")
            return

        print(f"Found {len(studies_df)} studies")

        # Set up paths
        temp_dir = collector.archive_processor.get_temp_dir()
        output_dir = Path("scripts/icpsr")
        output_dir.mkdir(exist_ok=True)

        print(f"Temp dir: {temp_dir}")
        print(f"Output dir: {output_dir}")

        # Test storage check
        print("Checking storage requirements...")
        storage_ok = collector.archive_processor.check_storage_requirements()
        print(f"Storage OK: {storage_ok}")

        # Test disk usage
        print("Getting disk usage...")
        total, used, free = get_disk_usage(str(temp_dir))
        print(f"Disk usage: total={total}, used={used}, free={free}")
        print(f"Types: {[type(x) for x in (total, used, free)]}")

        free_gb = float(free) / (1024**3)
        print(f"Free GB: {free_gb}")

        # Test ArchiveProgressLogger creation and storage logging
        print("Testing ArchiveProgressLogger...")
        logger = get_logger("debug_icpsr")

        print("Creating progress logger...")
        with ArchiveProgressLogger(
            logger, "Debug ICPSR collection", len(studies_df), show_size_info=True
        ) as archive_progress:
            print("Progress logger created")

            # Try logging storage info
            print("Logging storage info...")
            archive_progress.log_storage_info(free_gb)
            print("Storage info logged")

            # Try processing one study
            study = studies_df.iloc[0]
            study_id = study.get("study_id", "")
            print(f"Processing study: {study_id} (type: {type(study_id)})")

            # Test individual parts of download_and_process_study
            try:
                # Construct download URL
                download_url = (
                    f"https://www.openicpsr.org/openicpsr/project/{study_id}/version/V1/download/project"
                    f"?dirPath=/openicpsr/{study_id}/fcr:versions/V1"
                )
                print(f"Download URL: {download_url}")

                zip_file_path = temp_dir / f"ICPSR_{str(study_id).zfill(5)}.zip"
                extract_to = output_dir / study_id

                print(f"Zip file path: {zip_file_path} (type: {type(zip_file_path)})")
                print(f"Extract to: {extract_to} (type: {type(extract_to)})")

                # Test file size check
                if zip_file_path.exists():
                    file_size = zip_file_path.stat().st_size
                    print(f"File size: {file_size} (type: {type(file_size)})")

                    # Test progress update directly
                    print("Testing progress_logger.update_download...")
                    archive_progress.update_download(str(study_id), file_size)
                    print("update_download completed")

            except Exception as e:
                print(f"Detailed debug error: {e}")
                import traceback

                traceback.print_exc()

            valid_extensions = [".R", ".py", ".do", ".ipynb"]
            success, script_count = collector.download_and_process_study(
                study_id, temp_dir, output_dir, valid_extensions, archive_progress
            )

            print(f"Study result: success={success}, scripts={script_count}")

        print("Debug completed successfully!")

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    debug_icpsr()
