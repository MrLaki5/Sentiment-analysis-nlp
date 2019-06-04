import subprocess

STEM_OPTION_KESELJ_SIPKA_GREEDY = 1
STEM_OPTION_KESELJ_SIPKA_OPTIMAL = 2
STEM_OPTION_MILOSEVIC = 3
STEM_OPTION_LJUBESIC_PANDZIC = 4


# Use Serbian stemmers to stemm words from src_file
def stemm(stem_option, src_file_path, out_file_path):
    subprocess.call(['java', '-jar', 'SCStemmers.jar', str(stem_option), src_file_path, out_file_path])
