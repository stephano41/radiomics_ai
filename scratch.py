from src.inference import Inferrer

inferrer = Inferrer('8e6d214b085540b7a1ed70539f2ea977',
                    image_stems=('registered_adc', 't2', 'flair', 't1', 't1ce'),
                    mask_stem='mask',
                    extraction_config='./conf/radiomic_params/meningioma_mr.yaml')

print(inferrer.predict('data/test_meningioma_data'))