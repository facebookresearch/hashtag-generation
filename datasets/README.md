# Datasets

We use two datasets: Short-Form Video Dataset 1 (SFVD1) and Short-Form Video Dataset 2 (SFVD2).

Due to the requirements of the company's legal team, we are unable to provide dataset names.

# Data format
'''
{
    'id': data_id,
    'url': url,
    'video_file_path': video_file_path,
    'wave_path': audio_file_path,
    'frame_path': extracted_frame_files_path,
    'video_feature_path': frames_features_file_path,
    'audio_feature_path': audio_features_file_path,
    '50_constrains': retrieved_hashtags,
    'text': video_description,
    'hashtag': ['','',''],
}
'''
