import os
import sys
import numpy as np


def segment_videos(video_path, segment_file, output_folder):
    """
    This function will segment the videos downloaded from the MSP_Face corpus.
    This is a slightly modified code from MSP_Face Github 'download_and_segment_videos.py' file
    due to an issue with downloading the youtube videos.
    """
    video_name = os.path.basename(video_path)
    file_segments_data = np.genfromtxt(segment_file, dtype=[('f8'), ('f8'), ('S50')], delimiter='\t')

    if file_segments_data.shape == ():
        file_segments_data = np.atleast_1d(file_segments_data)

    gral_segments_path = os.path.join(output_folder, 'Segments')
    if not os.path.isdir(gral_segments_path):
        os.mkdir(gral_segments_path)

    for k in range(len(file_segments_data['f2'])):
        if video_name[:-4] in file_segments_data['f2'][k].decode('utf-8'):
            segments_path = os.path.join(gral_segments_path, file_segments_data['f2'][k].decode('utf-8')[:-4])
            if not os.path.isdir(segments_path):
                os.mkdir(segments_path)

            segment_output = os.path.join(segments_path, file_segments_data['f2'][k].decode('utf-8'))

            ti = file_segments_data['f0'][k]
            dt = file_segments_data['f1'][k] - file_segments_data['f0'][k]

            os.system(f"ffmpeg -loglevel panic -i {video_path} -strict -2 -ss {ti} -t {dt} {segment_output}")

            audio_path = os.path.join(segments_path, 'Audio')
            if not os.path.isdir(audio_path):
                os.mkdir(audio_path)

            audio_original = os.path.join(audio_path, file_segments_data['f2'][k].decode('utf-8')[:-4] + "O.wav")
            audio_stereo = os.path.join(audio_path, file_segments_data['f2'][k].decode('utf-8')[:-4] + "S.wav")
            audio_final = os.path.join(audio_path, file_segments_data['f2'][k].decode('utf-8')[:-4] + ".wav")

            os.system(f"ffmpeg -loglevel panic -i {segment_output} {audio_original}")
            os.system(f"sox {audio_original} {audio_stereo} remix 1")
            os.system(f"ffmpeg -loglevel panic -i {audio_stereo} -ac 1 -vn -acodec pcm_s16le -ar 16000 {audio_final}")

            os.remove(audio_original)
            os.remove(audio_stereo)

def main(videos_folder, segment_file, output_folder):
    for video_file in os.listdir(videos_folder):
        video_path = os.path.join(videos_folder, video_file)
        if os.path.isfile(video_path):
            segment_video(video_path, segment_file, output_folder)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: segment_videos.py <videos_folder> <segments_file> <output_folder>")
        sys.exit(1)
    
    videos_folder = sys.argv[1]
    segments_file = sys.argv[2]
    output_folder = sys.argv[3]

    main(videos_folder, segments_file, output_folder)
