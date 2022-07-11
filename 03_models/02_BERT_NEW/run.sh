#INPUT_DIR=C:/Users/siban/Dropbox/BICTOP/MyInvestor/06_model/02_NLP/05_spy_project_NEWS_v2/00_data/01_preprocessed
#OUTPUT_DIR=C:/Users/siban/Dropbox/BICTOP/MyInvestor/06_model/02_NLP/05_spy_project_NEWS_v2/00_data/02_runs/00_TEST

INPUT_DIR=/data/users/sibanez/03_MyInvestor_NEWS_v2/01_preprocessed
OUTPUT_DIR=/data/users/sibanez/03_MyInvestor_NEWS_v2/02_runs/00_TEST

MODEL_FILENAME=model_v0.py

python -m ipdb train_test.py \
    --input_dir=$INPUT_DIR \
    --output_dir=$OUTPUT_DIR \
    --model_filename=$MODEL_FILENAME \
    --task=Train \
    \
    --model_name=ProsusAI/finbert \
    --seq_len=256 \
    --n_labels=3 \
    --n_heads=8 \
    --hidden_dim=768 \
    --freeze_BERT=True \
    --seed=1234 \
    --use_cuda=True \
    \
    --n_epochs=10 \
    --batch_size_train=50 \
    --shuffle_train=False \
    --drop_last_train=False \
    --dev_train_ratio=1 \
    --train_toy_data=True \
    --len_train_toy_data=100/data/users/sibanez/03_MyInvestor_NEWS_v2/01_preprocessed \
    --lr=2e-5 \
    --wd=1e-6 \
    --dropout=0.2 \
    --momentum=0.9 \
    --save_final_model=True \
    --save_model_steps=True \
    --save_step_cliff=0 \
    --gpu_ids_train=0 \
    \
    --test_file=model_test.pkl \
    --model_file=model.pt.40 \
    --batch_size_test=100 \
    --gpu_id_test=0 \

#read -p 'EOF'

#--model_name=nlpaueb/legal-bert-small-uncased \
#--hidden_dim=512 \

#--task=Train / Test
#--pooing=Avg / Max
#--batch_size=280 / 0,1,2,3
#--wd=1e-6
