"""
Speaker Identification Streamlit App
Integrates with speaker_id.py for training and testing speaker recognition models

File Paths:
- Training audio: training_set/
- Test audio: testing_set/
- Models: trained_models/
- Manifests: training_set_addition.txt, testing_set_addition.txt

Unknown Speaker Threshold: -20 (hard-coded in speaker_id.py)

Run: streamlit run app.py
"""

import streamlit as st
import os
import numpy as np
import pandas as pd
from scipy.io.wavfile import read
import SpeakerIdentification as sid
from datetime import datetime

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Speaker Identification System",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== SIDEBAR ====================
st.sidebar.title("🎤 Speaker ID System")
st.sidebar.markdown("---")
st.sidebar.write("""
**Train** a Gaussian Mixture Model (GMM) on speaker audio samples, then **test** on new audio to identify speakers or mark as UNKNOWN.

- **Unknown Threshold:** -20 (hard-coded)
- **Training files:** `training_set/`
- **Test files:** `testing_set/`
- **Models:** `trained_models/`
""")

st.sidebar.markdown("---")
mode = st.sidebar.radio("**Select Mode:**", ["🎓 Train", "🧪 Test"])

# ==================== HELPER FUNCTIONS ====================


def format_filesize(size_bytes):
    """Format bytes to human readable size"""
    for unit in ['B', 'KB', 'MB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f}{unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f}GB"


def get_audio_duration(filepath):
    """Get duration of audio file in seconds"""
    try:
        sr, audio = read(filepath)
        duration = len(audio) / sr
        return duration
    except:
        return 0


def play_audio(filepath):
    """Display audio player"""
    try:
        with open(filepath, 'rb') as f:
            audio_data = f.read()
        st.audio(audio_data, format='audio/wav')
    except Exception as e:
        st.error(f"Could not load audio: {e}")


# ==================== TRAIN TAB ====================
if mode == "🎓 Train":
    st.title("🎓 Train Speaker Models")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### Input Audio")
        speaker_name = st.text_input(
            "👤 Speaker Name",
            placeholder="e.g., Haritha, John, Alice",
            help="Name of the speaker for training"
        )

    with col2:
        st.markdown("### Recording Options")
        record_duration = st.slider("Duration (seconds)", 5, 30, 10)

    # Recording and Upload Section
    st.markdown("---")
    rec_col1, rec_col2 = st.columns(2)

    with rec_col1:
        st.subheader("📝 Record Audio")
        if st.button("🎤 Record Training Sample", key="record_train", use_container_width=True):
            if speaker_name:
                try:
                    with st.spinner(f"Recording {record_duration}s..."):
                        filepath = sid.record_audio_train(
                            speaker_name,
                            duration=record_duration
                        )
                        if filepath:
                            st.success(
                                f"✓ Recorded: {os.path.basename(filepath)}")
                            st.balloons()
                        else:
                            st.error("Failed to record audio")
                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.warning("Please enter a speaker name")

    with rec_col2:
        st.subheader("📤 Upload Audio")
        uploaded_file = st.file_uploader(
            "Upload WAV file",
            type=['wav'],
            key="upload_train"
        )
        if uploaded_file and speaker_name:
            if st.button("✅ Save Uploaded File", key="save_upload_train", use_container_width=True):
                filepath = sid.save_uploaded_train(
                    uploaded_file.read(),
                    uploaded_file.name,
                    speaker_name
                )
                if filepath:
                    st.success(f"✓ Saved: {os.path.basename(filepath)}")
                    st.balloons()
                else:
                    st.error("Failed to save file")
        elif uploaded_file:
            st.warning("Please enter a speaker name first")

    # Training Files Table
    st.markdown("---")
    st.subheader("📋 Training Files")

    training_files = sid.get_training_files()

    if training_files:
        # Display by speaker
        speakers_dict = {}
        for file_info in training_files:
            speaker = file_info['speaker']
            if speaker not in speakers_dict:
                speakers_dict[speaker] = []
            speakers_dict[speaker].append(file_info)

        # Summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Files", len(training_files))
        with col2:
            st.metric("Speakers", len(speakers_dict))
        with col3:
            total_size = sum(f['size'] for f in training_files)
            st.metric("Total Size", format_filesize(total_size))

        st.markdown("---")

        # Files per speaker - FIXED: Unique keys for each widget
        speaker_file_counter = 0
        for speaker, files in sorted(speakers_dict.items()):
            with st.expander(f"**{speaker}** ({len(files)} files)", expanded=True):
                # Create dataframe for display
                df_data = []
                for f in files:
                    duration = get_audio_duration(f['filepath'])
                    df_data.append({
                        "Filename": f['filename'],
                        "Duration": f"{duration:.1f}s",
                        "Size": format_filesize(f['size'])
                    })

                # Display dataframe
                st.dataframe(pd.DataFrame(df_data), use_container_width=True)

                # Playback and delete - FIXED: Using unique keys
                st.write("")  # Add spacing
                for i, f in enumerate(files):
                    col1, col2, col3 = st.columns([2, 1, 1])
                    with col1:
                        st.write(f"**{f['filename']}**")
                    with col2:
                        # FIXED: Unique key using speaker + filename
                        if st.button("▶️ Play", key=f"play_{speaker}_{i}_{f['filename']}", use_container_width=True):
                            play_audio(f['filepath'])
                    with col3:
                        # FIXED: Unique key using speaker + filename
                        if st.button("🗑️ Delete", key=f"del_{speaker}_{i}_{f['filename']}", use_container_width=True):
                            sid.delete_training_file(f['filename'])
                            st.success("✓ Deleted")
                            st.experimental_rerun()

        # Train Button
        st.markdown("---")
        st.subheader("🚀 Train Model")

        if st.button("🚀 Train All Speakers", key="train_models", use_container_width=True, type="primary"):
            status_container = st.empty()

            with status_container.container():
                st.info("⏳ Training in progress...")

            # Train
            summary = sid.train_models()

            if "error" in summary:
                st.error(f"❌ Training failed: {summary['error']}")
            else:
                status_container.empty()
                st.success("✓ Training completed!")

                # Display summary
                if summary.get("speakers"):
                    st.subheader("📊 Training Summary")
                    summary_data = []
                    for speaker, info in summary["speakers"].items():
                        summary_data.append({
                            "Speaker": speaker,
                            "Samples": info.get("samples", 0),
                            "Model": info.get("model_file", "N/A")
                        })

                    st.dataframe(pd.DataFrame(summary_data),
                                 use_container_width=True)
    else:
        st.info("📌 No training files yet. Record or upload audio to get started.")
        st.warning("⚠️ **Train button will appear once files are present**")

# ==================== TEST TAB ====================
else:  # Test mode
    st.title("🧪 Test Speaker Identification")

    # Check if models exist
    trained_speakers = sid.get_trained_speakers()

    if not trained_speakers:
        st.warning(
            "⚠️ No trained models found. Please train models first in the Train tab.")
    else:
        st.success(f"✓ Trained speakers: {', '.join(trained_speakers)}")

    st.markdown("---")

    # Recording and Upload Section
    test_col1, test_col2 = st.columns(2)

    with test_col1:
        st.subheader("📝 Record Test Audio")
        test_label = st.text_input(
            "Test Label",
            placeholder="e.g., test, unknown_person",
            value="test"
        )
        if st.button("🎤 Record Test Sample", key="record_test", use_container_width=True):
            try:
                with st.spinner("Recording 10s..."):
                    filepath = sid.record_audio_test(test_label, duration=10)
                    if filepath:
                        st.success(f"✓ Recorded: {os.path.basename(filepath)}")
                        st.balloons()
                    else:
                        st.error("Failed to record")
            except Exception as e:
                st.error(f"Error: {e}")

    with test_col2:
        st.subheader("📤 Upload Test Audio")
        uploaded_test = st.file_uploader(
            "Upload WAV file",
            type=['wav'],
            key="upload_test"
        )
        if uploaded_test and st.button("✅ Save & Test", key="save_test", use_container_width=True):
            filepath = sid.save_uploaded_test(
                uploaded_test.read(),
                uploaded_test.name,
                "uploaded"
            )
            if filepath:
                st.success(f"✓ Saved: {os.path.basename(filepath)}")
                st.balloons()
            else:
                st.error("Failed to save")

    st.markdown("---")

    # Testing options
    st.subheader("🧪 Test Options")
    test_option = st.radio(
        "How would you like to test?",
        ["Run All Tests", "Select Specific Files"]
    )

    testing_files = sid.get_testing_files()

    if not testing_files:
        st.info("📌 No test files yet. Record or upload audio.")
    else:
        if test_option == "Run All Tests":
            if st.button("🔍 Run All Tests", key="run_all_tests", type="primary", use_container_width=True):
                if not trained_speakers:
                    st.error("No trained models available")
                else:
                    with st.spinner("Testing all files..."):
                        results = sid.test_all_from_list(
                            threshold=sid.UNKNOWN_THRESHOLD)

                    if results:
                        # Display results
                        st.subheader("📊 Test Results")

                        result_data = []
                        unknown_count = 0
                        identified_count = 0

                        for filename, speaker, score in results:
                            result_data.append({
                                "File": filename,
                                "Predicted": speaker,
                                "Score": f"{score:.2f}"
                            })

                            if speaker == "UNKNOWN SPEAKER":
                                unknown_count += 1
                            else:
                                identified_count += 1

                        # Summary metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Tests", len(results))
                        with col2:
                            st.metric("Identified", identified_count)
                        with col3:
                            st.metric("Unknown", unknown_count)

                        st.markdown("---")

                        # Results table
                        st.dataframe(pd.DataFrame(result_data),
                                     use_container_width=True)

                        # Download CSV
                        csv = pd.DataFrame(result_data).to_csv(index=False)
                        st.download_button(
                            "📥 Download Results (CSV)",
                            csv,
                            f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            "text/csv",
                            use_container_width=True
                        )

        else:  # Select Specific Files
            st.write("**Select files to test:**")
            selected_files = st.multiselect(
                "Choose test files",
                [f['filename'] for f in testing_files],
                key="file_select"
            )

            if selected_files and st.button("🔍 Test Selected", key="test_selected", type="primary", use_container_width=True):
                if not trained_speakers:
                    st.error("No trained models available")
                else:
                    result_data = []
                    unknown_count = 0

                    with st.spinner(f"Testing {len(selected_files)} files..."):
                        for filename in selected_files:
                            filepath = os.path.join(
                                sid.TESTING_FOLDER, filename)
                            if os.path.exists(filepath):
                                speaker, score = sid.test_model(
                                    filepath, sid.UNKNOWN_THRESHOLD)
                                result_data.append({
                                    "File": filename,
                                    "Predicted": speaker,
                                    "Score": f"{score:.2f}"
                                })
                                if speaker == "UNKNOWN SPEAKER":
                                    unknown_count += 1
                            else:
                                st.warning(f"File not found: {filename}")

                    if result_data:
                        st.subheader("📊 Test Results")
                        st.dataframe(pd.DataFrame(result_data),
                                     use_container_width=True)

                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Tested", len(result_data))
                        with col2:
                            st.metric("Unknown", unknown_count)

# ==================== FOOTER ====================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 12px;'>
Speaker Identification System | Built with Streamlit | Threshold: -20
</div>
""", unsafe_allow_html=True)
