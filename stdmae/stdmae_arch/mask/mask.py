import torch
from torch import nn
from timm.models.vision_transformer import trunc_normal_

from .patch import PatchEmbedding
from .maskgenerator import MaskGenerator
from .positional_encoding import PositionalEncoding
from .transformer_layers import TransformerLayers


def unshuffle(shuffled_tokens):
    dic = {}
    for k, v, in enumerate(shuffled_tokens):
        dic[v] = k
    unshuffle_index = []
    for i in range(len(shuffled_tokens)):
        unshuffle_index.append(dic[i])
    return unshuffle_index


class Mask(nn.Module):
    """
    STD-MAE(Spatial-Temporal-Decoupled Masked AutoEncoder)의 핵심 구현 클래스
    시간적(Temporal) 또는 공간적(Spatial) 마스킹 및 재구성을 담당
    """

    def __init__(self, patch_size, in_channel, embed_dim, num_heads, mlp_ratio, dropout, mask_ratio, encoder_depth, decoder_depth, spatial=False, mode="pre-train"):
        super().__init__()
        assert mode in ["pre-train", "forecasting"], "Error mode."
        self.patch_size = patch_size      # 각 패치의 시간 길이 (논문에서 L)
        self.in_channel = in_channel      # 입력 채널 수 (논문에서 C)
        self.embed_dim = embed_dim        # 임베딩 차원 (논문에서 d)
        self.num_heads = num_heads        # 트랜스포머의 어텐션 헤드 수
        self.mask_ratio = mask_ratio      # 마스킹 비율 (논문에서 r)
        self.encoder_depth = encoder_depth  # 인코더 트랜스포머 레이어 수
        self.mode = mode                  # 실행 모드: 사전학습 또는 예측
        self.mlp_ratio = mlp_ratio        # 트랜스포머 MLP의 확장 비율
        self.spatial = spatial            # 공간 모드(True) 또는 시간 모드(False) 설정
        self.selected_feature = 0         # 사용할 특성 인덱스

        # 정규화 레이어
        self.encoder_norm = nn.LayerNorm(embed_dim)
        self.decoder_norm = nn.LayerNorm(embed_dim)
        self.pos_mat = None               # 위치 인코딩 행렬 저장 변수
        
        # 인코더 구성요소
        # 패치 임베딩 - 시계열을 고정 크기 패치로 분할하고 임베딩
        self.patch_embedding = PatchEmbedding(patch_size, in_channel, embed_dim, norm_layer=None)
        # 위치 인코딩 - 패치와 노드에 위치 정보 추가
        self.positional_encoding = PositionalEncoding()

        # 인코더 트랜스포머 레이어
        self.encoder = TransformerLayers(embed_dim, encoder_depth, mlp_ratio, num_heads, dropout)

        # 디코더 구성요소
        # 인코더에서 디코더로의 변환 레이어
        self.enc_2_dec_emb = nn.Linear(embed_dim, embed_dim, bias=True)
        # 마스크 토큰 - 마스킹된 위치에 삽입할 학습 가능한 토큰
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))

        # 디코더 트랜스포머 레이어
        self.decoder = TransformerLayers(embed_dim, decoder_depth, mlp_ratio, num_heads, dropout)

        # 출력(재구성) 레이어
        self.output_layer = nn.Linear(embed_dim, patch_size)
        self.initialize_weights()

    def initialize_weights(self):
        """마스크 토큰의 가중치 초기화"""
        trunc_normal_(self.mask_token, std=.02)

    def encoding(self, long_term_history, mask=True):
        """인코딩 과정: 입력 데이터를 패치화하고 마스킹 후 인코딩

        Args:
            long_term_history (torch.Tensor): 장기 시계열 데이터 [B, N, C, P * L]
                                              B: 배치 크기, N: 노드 수, C: 채널, P: 패치 수, L: 패치 크기
            mask (bool): 마스킹 여부 (사전학습에서는 True, 예측에서는 False)

        Returns:
            torch.Tensor: 마스킹되지 않은 토큰의 표현(Temporal/Spatial Representation)
            list: 마스킹되지 않은 토큰 인덱스
            list: 마스킹된 토큰 인덱스
        """

        # 패치화 및 임베딩
        if mask:  # 사전학습 모드 (마스킹 사용)
            if self.spatial:  # 공간 마스킹 모드
                # 1.1 Patch Embedding: 시계열을 패치로 분할 및 임베딩
                patches = self.patch_embedding(long_term_history)  # B, N, d, P
                patches = patches.transpose(-1, -2)  # B, N, P, d
                batch_size, num_nodes, num_time, num_dim = patches.shape
                
                # 1.2 S-T Positional Encoding: 시공간 위치 정보 추가
                patches, self.pos_mat = self.positional_encoding(patches)
                
                # 1.3 S-Mask: 공간 차원(노드)에서 무작위 마스킹
                Maskg = MaskGenerator(patches.shape[1], self.mask_ratio)
                unmasked_token_index, masked_token_index = Maskg.uniform_rand()
                encoder_input = patches[:, unmasked_token_index, :, :]
                encoder_input = encoder_input.transpose(-2, -3)
                
                # 1.4 Transformer Layers: 마스킹되지 않은 노드에서 표현 학습
                hidden_states_unmasked = self.encoder(encoder_input)
                # 정규화 및 차원 재조정
                hidden_states_unmasked = self.encoder_norm(hidden_states_unmasked).view(batch_size, num_time, -1, self.embed_dim)
                # 결과: 3.5 Spatial Representation (공간적 표현)

            if not self.spatial:  # 시간 마스킹 모드
                # 1.1 Patch Embedding: 시계열을 패치로 분할 및 임베딩
                patches = self.patch_embedding(long_term_history)  # B, N, d, P
                patches = patches.transpose(-1, -2)  # B, N, P, d
                batch_size, num_nodes, num_time, num_dim = patches.shape

                # 1.2 S-T Positional Encoding: 시공간 위치 정보 추가
                patches, self.pos_mat = self.positional_encoding(patches)
                
                # 1.3 T-Mask: 시간 차원(패치)에서 무작위 마스킹
                Maskg = MaskGenerator(patches.shape[2], self.mask_ratio)
                unmasked_token_index, masked_token_index = Maskg.uniform_rand()
                encoder_input = patches[:, :, unmasked_token_index, :]
                
                # 1.4 Transformer Layers: 마스킹되지 않은 패치에서 표현 학습
                hidden_states_unmasked = self.encoder(encoder_input)
                # 정규화 및 차원 재조정
                hidden_states_unmasked = self.encoder_norm(hidden_states_unmasked).view(batch_size, num_nodes, -1, self.embed_dim)
                # 결과: 1.5 Temporal Representation (시간적 표현)

        else:  # 예측 모드 (마스킹 없음)
            batch_size, num_nodes, _, _ = long_term_history.shape
            # 패치화 및 임베딩
            patches = self.patch_embedding(long_term_history)     # B, N, d, P
            patches = patches.transpose(-1, -2)         # B, N, P, d
            # 위치 인코딩
            patches, self.pos_mat = self.positional_encoding(patches)  # B, N, P, d
            
            unmasked_token_index, masked_token_index = None, None
            encoder_input = patches  # B, N, P, d
            if self.spatial:
                encoder_input = encoder_input.transpose(-2, -3)  # B, P, N, d
            
            # 트랜스포머 레이어 처리
            hidden_states_unmasked = self.encoder(encoder_input)  # B, P, N, d 또는 B, N, P, d
            if self.spatial:
                hidden_states_unmasked = hidden_states_unmasked.transpose(-2, -3)  # B, N, P, d
            
            # 정규화 및 차원 재조정
            hidden_states_unmasked = self.encoder_norm(hidden_states_unmasked).view(batch_size, num_nodes, -1, self.embed_dim)  # B, N, P, d
            return hidden_states_unmasked, unmasked_token_index, masked_token_index

        return hidden_states_unmasked, unmasked_token_index, masked_token_index

    def decoding(self, hidden_states_unmasked, masked_token_index):
        """디코딩 과정: 마스킹된 토큰 재구성

        Args:
            hidden_states_unmasked: 인코더에서 얻은 마스킹되지 않은 토큰의 표현 (Temporal/Spatial Representation)
            masked_token_index: 마스킹된 토큰의 인덱스

        Returns:
            torch.Tensor: 재구성된 전체 시퀀스 (마스킹된 토큰 포함)
        """

        # 인코더 출력을 디코더 입력으로 변환
        hidden_states_unmasked = self.enc_2_dec_emb(hidden_states_unmasked)  # B, N, P, d 또는 B, P, N, d
        
        if self.spatial:  # 공간 디코더
            # 배치, 시간, 노드 수 및 차원 정보 추출
            batch_size, num_time, num_nodes, _ = hidden_states_unmasked.shape  # B, P, N, d
            
            # 4.1 Spatial Padding: 마스킹되지 않은 노드 인덱스 계산
            unmasked_token_index = [i for i in range(0, len(masked_token_index) + num_nodes) if i not in masked_token_index]
            
            # 4.2 S-T Positional Encoding: 마스킹된 노드에 대한 위치 인코딩 가져오기
            hidden_states_masked = self.pos_mat[:, masked_token_index, :, :]  # B, N*r, P, d
            hidden_states_masked = hidden_states_masked.transpose(-2, -3)  # B, P, N*r, d

            # 4.3 Mask Token: 마스킹된 위치에 학습 가능한 마스크 토큰 추가
            hidden_states_masked += self.mask_token.expand(batch_size, num_time, len(masked_token_index), hidden_states_unmasked.shape[-1])  # B, P, N*r, d
            
            # 마스킹되지 않은 토큰에 위치 인코딩 추가
            hidden_states_unmasked += self.pos_mat[:, unmasked_token_index, :, :].transpose(-2, -3)  # B, P, N*(1-r), d
            
            # 마스킹된 토큰과 마스킹되지 않은 토큰 결합
            hidden_states_full = torch.cat([hidden_states_unmasked, hidden_states_masked], dim=-2)   # B, P, N, d

            # 4.4 Transformer Layers: 디코더 트랜스포머 처리
            hidden_states_full = self.decoder(hidden_states_full)  # B, P, N, d
            hidden_states_full = self.decoder_norm(hidden_states_full)  # B, P, N, d
            
            # 4.5 Spatial Reconstruction: 출력 레이어로 마스킹된 노드 재구성
            reconstruction_full = self.output_layer(hidden_states_full.view(batch_size, num_time, -1, self.embed_dim))  # B, P, N, L
        
        else:  # 시간 디코더
            # 배치, 노드, 시간 및 차원 정보 추출
            batch_size, num_nodes, num_time, _ = hidden_states_unmasked.shape
            
            # 2.1 Temporal Padding: 마스킹되지 않은 시간 패치 인덱스 계산
            # 이 과정에서 마스킹된 토큰과 마스킹되지 않은 토큰의 위치를 파악하고 전체 시퀀스 복원 프레임워크 구성
            unmasked_token_index = [i for i in range(0, len(masked_token_index) + num_time) if i not in masked_token_index]
            
            # 2.2 S-T Positional Encoding: 마스킹된 위치에 대한 위치 인코딩 가져오기
            hidden_states_masked = self.pos_mat[:, :, masked_token_index, :]
            
            # 2.3 Mask Token: 마스킹된 위치에 학습 가능한 마스크 토큰 추가
            # 여기서 위치 인코딩과 마스크 토큰이 결합됨
            hidden_states_masked += self.mask_token.expand(batch_size, num_nodes, len(masked_token_index), hidden_states_unmasked.shape[-1])
            
            # 마스킹되지 않은 토큰에 위치 인코딩 추가
            hidden_states_unmasked += self.pos_mat[:, :, unmasked_token_index, :]
            
            # 마스킹된 토큰과 마스킹되지 않은 토큰 결합 (전체 시퀀스 구성)
            hidden_states_full = torch.cat([hidden_states_unmasked, hidden_states_masked], dim=-2)   # B, N, P, d

            # 2.4 Transformer Layers: 디코더 트랜스포머 처리
            hidden_states_full = self.decoder(hidden_states_full)
            hidden_states_full = self.decoder_norm(hidden_states_full)

            # 2.5 Temporal Reconstruction: 출력 레이어로 마스킹된 패치 재구성
            reconstruction_full = self.output_layer(hidden_states_full.view(batch_size, num_nodes, -1, self.embed_dim))

        return reconstruction_full

    def get_reconstructed_masked_tokens(self, reconstruction_full, real_value_full, unmasked_token_index,
                                        masked_token_index):
        """재구성된 마스킹 토큰과 해당 실제 값을 추출하여 손실 계산에 사용

        Args:
            reconstruction_full (torch.Tensor): 재구성된 전체 토큰
            real_value_full (torch.Tensor): 실제 전체 토큰 값
            unmasked_token_index (list): 마스킹되지 않은 토큰 인덱스
            masked_token_index (list): 마스킹된 토큰 인덱스

        Returns:
            torch.Tensor: 재구성된 마스킹 토큰
            torch.Tensor: 실제 마스킹 토큰 값
        """
        # 재구성된 마스킹 토큰 추출
        if self.spatial:  # 공간 모드
            batch_size, num_time, num_nodes, _ = reconstruction_full.shape  # B, P, N, L
            # 재구성된 마스킹 토큰만 추출 (마스킹되지 않은 토큰 이후 부분)
            reconstruction_masked_tokens = reconstruction_full[:, :, len(unmasked_token_index):, :]  # B, P, r*N, L
            reconstruction_masked_tokens = reconstruction_masked_tokens.view(batch_size, num_time, -1)  # B, P, r*N*L
    
            # 원본 데이터에서 마스킹된 토큰에 해당하는 실제 값 추출
            label_full = real_value_full.permute(0, 3, 1, 2).unfold(1, self.patch_size, self.patch_size)[:, :, :, self.selected_feature, :].transpose(1, 2)  # B, N, P, L
            label_masked_tokens = label_full[:, masked_token_index, :, :].transpose(1, 2).contiguous()  # B, P, N*r, L
            label_masked_tokens = label_masked_tokens.view(batch_size, num_time, -1)  # B, P, r*N*L
    
            return reconstruction_masked_tokens, label_masked_tokens
        else:  # 시간 모드
            batch_size, num_nodes, num_time, _ = reconstruction_full.shape

            # 재구성된 마스킹 토큰만 추출
            reconstruction_masked_tokens = reconstruction_full[:, :, len(unmasked_token_index):, :]  # B, N, r*P, d
            reconstruction_masked_tokens = reconstruction_masked_tokens.view(batch_size, num_nodes, -1).transpose(1, 2)  # B, r*P*d, N

            # 원본 데이터에서 마스킹된 토큰에 해당하는 실제 값 추출
            label_full = real_value_full.permute(0, 3, 1, 2).unfold(1, self.patch_size, self.patch_size)[:, :, :, self.selected_feature, :].transpose(1, 2)  # B, N, P, L
            label_masked_tokens = label_full[:, :, masked_token_index, :].contiguous()  # B, N, r*P, d
            label_masked_tokens = label_masked_tokens.view(batch_size, num_nodes, -1).transpose(1, 2)  # B, r*P*d, N

            return reconstruction_masked_tokens, label_masked_tokens

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor = None, batch_seen: int = None, epoch: int = None, **kwargs) -> torch.Tensor:
        """모델의 순전파 과정

        Args:
            history_data: 과거 시계열 데이터
            future_data: 미래 시계열 데이터 (예측 모드에서 사용)
            batch_seen: 현재까지 본 배치 수
            epoch: 현재 에폭

        Returns:
            사전학습 모드: (재구성된 마스킹 토큰, 실제 마스킹 토큰 값)
            예측 모드: 마스킹 없이 인코딩된 전체 표현
        """
        # 입력 데이터 차원 변환
        history_data = history_data.permute(0, 2, 3, 1)  # B, N, 1, L * P
        
        # 실행 모드에 따른 처리
        if self.mode == "pre-train":  # 사전학습 모드
            # 1. 인코딩: 마스킹 및 인코더 처리
            hidden_states_unmasked, unmasked_token_index, masked_token_index = self.encoding(history_data)
            
            # 2. 디코딩: 마스킹된 토큰 재구성
            reconstruction_full = self.decoding(hidden_states_unmasked, masked_token_index)
            
            # 3. 손실 계산을 위한 마스킹 토큰 추출
            reconstruction_masked_tokens, label_masked_tokens = self.get_reconstructed_masked_tokens(
                reconstruction_full, history_data, unmasked_token_index, masked_token_index)

            return reconstruction_masked_tokens, label_masked_tokens
        else:  # 예측 모드
            # 마스킹 없이 전체 시퀀스 인코딩
            hidden_states_full, _, _ = self.encoding(history_data, mask=False)
            return hidden_states_full

def main():
    import sys
    from torchsummary import summary
    GPU = sys.argv[-1] if len(sys.argv) == 2 else '2'
    device = torch.device("cuda:{}".format(GPU)) if torch.cuda.is_available() else torch.device("cpu")
    model = Mask(
    patch_size=12,
    in_channel=1,
    embed_dim=96,
    num_heads=4,
    mlp_ratio=4,
    dropout=0.1,
    mask_ratio=0.75,
    encoder_depth=4,
    decoder_depth=1,
    mode="pre-train"
).to(device)
    summary(model, (288*7, 307, 1), device=device)


if __name__ == '__main__':
    main()
