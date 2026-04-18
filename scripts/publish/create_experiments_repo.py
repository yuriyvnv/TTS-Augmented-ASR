from dotenv import load_dotenv
load_dotenv()
from huggingface_hub import HfApi
api = HfApi()                                                                                                                        
api.create_repo('yuriyvnv/experiments_parakeet', repo_type='model', exist_ok=True)                                                 
api.upload_folder(
folder_path='results/parakeet_finetune_et/cv_synth_all_et_s42',
repo_id='yuriyvnv/experiments_parakeet',
path_in_repo='parakeet-tdt-cv_synth_all_et-seed42',
commit_message='Upload parakeet-tdt-cv_synth_all_et-seed42',
ignore_patterns=['data/*'],
)
print('Done!')