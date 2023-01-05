import pandas as pd
from src.common.constants import BLOCK_SIZE, COL_WORD_FRAME_SELECTED


TOKEN_CLS    = '[CLS]'
TOKEN_CLS_ID = 101
TOKEN_SEP    = '[SEP]'
TOKEN_SEP_ID = 102
TOKEN_PAD    = '[PAD]'
TOKEN_PAD_ID = 0


def pad_tokens(df_tokens, interval_id, video_id):
    all_df_blocks_padded = []
    for i in range(0, df_tokens.shape[0], BLOCK_SIZE):
        df_tokens_block = df_tokens[i:i + BLOCK_SIZE]
        df_tokens_block_padded = pad_tokens_block(df_tokens_block, interval_id, video_id)
        all_df_blocks_padded.append(df_tokens_block_padded)
    df_tokens_padded = pd.concat(all_df_blocks_padded)
    if 0 < i:
        print(f'Long interval {interval_id} ({video_id}), total sequence: {len(df_tokens_padded)}')
    return df_tokens_padded


def pad_tokens_block(df_tokens, interval_id, video_id):
    padding_count = BLOCK_SIZE - len(df_tokens)
    df_tokens_padded = pd.concat([
        df_cls_token(interval_id, video_id),
        df_tokens,
        df_sep_token(interval_id, video_id),
        df_pad_token(interval_id, video_id, padding_count)
    ])
    assert len(df_tokens_padded) == (BLOCK_SIZE + 2) # 128
    return df_tokens_padded


def df_cls_token(interval_id, video_id):
    cls_row = {'word': TOKEN_CLS, 'bert_token': TOKEN_CLS, 'token_id': TOKEN_CLS_ID,
               'interval_id': interval_id, 'video_id': video_id,
               COL_WORD_FRAME_SELECTED: -1, 'start_frame': -1, 'end_frame': -1, 'start_char': -1, 'end_char': -1,
               'selected_frame_fix': -1, 'voken': None}
    return pd.DataFrame.from_records([cls_row])


def df_sep_token(interval_id, video_id):
    sep_row = {'word': TOKEN_SEP, 'bert_token': TOKEN_SEP, 'token_id': TOKEN_SEP_ID,
               'interval_id': interval_id, 'video_id': video_id,
               COL_WORD_FRAME_SELECTED: -1, 'start_frame': -1, 'end_frame': -1, 'start_char': -1, 'end_char': -1,
               'selected_frame_fix': -1, 'voken': None}
    return pd.DataFrame.from_records([sep_row])


def df_pad_token(interval_id, video_id, padding_count):
    pad_row = {'word': TOKEN_PAD, 'bert_token': TOKEN_PAD, 'token_id': TOKEN_PAD_ID,
               'interval_id': interval_id, 'video_id': video_id,
               COL_WORD_FRAME_SELECTED: -1, 'start_frame': -1, 'end_frame': -1, 'start_char': -1, 'end_char': -1,
               'selected_frame_fix': -1, 'voken': None}
    return pd.DataFrame.from_records(padding_count * [pad_row])
