import torch
from torch import nn
from .mask import Mask
from .graphwavenet import GraphWaveNet

class STDMAE(nn.Module):
    """
    Spatio-Temporal-Decoupled Masked Pre-training for Traffic Forecasting
    
    STD-MAE 모델은 시공간 디커플링 마스킹 자기지도학습(self-supervised learning)을 통해
    시계열 예측 성능을 향상시키는 프레임워크입니다.
    """
    def __init__(self, dataset_name, pre_trained_tmae_path, pre_trained_smae_path, mask_args, backend_args):
        """
        STD-MAE 모델 초기화
        
        Args:
            dataset_name (str): 데이터셋 이름 (예: 'PEMS04', 'PEMS08')
            pre_trained_tmae_path (str): 사전학습된 시간(Temporal) MAE 모델 경로
            pre_trained_smae_path (str): 사전학습된 공간(Spatial) MAE 모델 경로
            mask_args (dict): Mask 클래스 초기화 인자
            backend_args (dict): 백엔드 모델(GraphWaveNet) 초기화 인자
        """
        super().__init__()
        self.dataset_name = dataset_name
        self.pre_trained_tmae_path = pre_trained_tmae_path
        self.pre_trained_smae_path = pre_trained_smae_path
        
        # 모델 구성요소 초기화
        # 1. 시간적 마스킹 인코더(T-MAE)
        self.tmae = Mask(**mask_args)
        
        # 2. 공간적 마스킹 인코더(S-MAE)
        self.smae = Mask(spatial=True, **mask_args)  # 공간 마스킹 모드 활성화
        
        # 3. 백엔드 예측 모델(GraphWaveNet)
        self.backend = GraphWaveNet(**backend_args)
        
        # 사전학습 모델 로드
        self.load_pre_trained_model()
    
    def load_pre_trained_model(self):
        """
        사전학습된 TMAE 및 SMAE 모델 로드 및 파라미터 고정
        """
        # TMAE 모델 파라미터 로드
        checkpoint_dict = torch.load(self.pre_trained_tmae_path)
        self.tmae.load_state_dict(checkpoint_dict["model_state_dict"])
        
        # SMAE 모델 파라미터 로드
        checkpoint_dict = torch.load(self.pre_trained_smae_path)
        self.smae.load_state_dict(checkpoint_dict["model_state_dict"])
        
        # 인코더 파라미터 고정 (fine-tuning 단계에서 학습되지 않도록)
        for param in self.tmae.parameters():
            param.requires_grad = False
        for param in self.smae.parameters():
            param.requires_grad = False
    
    def forward(self, history_data: torch.Tensor, long_history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, **kwargs) -> torch.Tensor:
        """
        STD-MAE 모델의 순전파
        
        Args:
            history_data (torch.Tensor): 단기 이력 데이터, 형태: [B, L, N, C]
                                         B: 배치 크기, L: 입력 시퀀스 길이(보통 12),
                                         N: 노드 수, C: 채널 수(보통 1 또는 3)
            long_history_data (torch.Tensor): 장기 이력 데이터, 형태: [B, L_long, N, C]
                                             L_long: 장기 입력 길이(보통 L * P, 예: 12 * 168 = 2016)
            future_data (torch.Tensor): 미래 데이터(학습 시 사용), 형태: [B, L_pred, N, C]
            batch_seen (int): 지금까지 처리한 배치 수
            epoch (int): 현재 에폭
        
        Returns:
            torch.Tensor: 예측 결과, 형태: [B, L_pred, N, C]
        """
        # 입력 데이터 준비
        short_term_history = history_data     # 단기 이력 데이터 [B, L, N, C]
        batch_size, _, num_nodes, _ = history_data.shape
        
        # 1. 시간적 표현 추출 (TMAE를 통한 인코딩)
        # long_history_data[..., [0]]는 첫 번째 채널만 선택 (보통 교통 속도)
        hidden_states_t = self.tmae(long_history_data[..., [0]])  # [B, N, P, d]
        
        # 2. 공간적 표현 추출 (SMAE를 통한 인코딩)
        hidden_states_s = self.smae(long_history_data[..., [0]])  # [B, N, P, d]
        
        # 3. 시간적 및 공간적 표현 결합
        hidden_states = torch.cat((hidden_states_t, hidden_states_s), -1)  # [B, N, P, 2d]
        
        # 4. 최근 패치 선택 (마지막 패치만 사용)
        out_len = 1  # 선택할 패치 수
        hidden_states = hidden_states[:, :, -out_len, :]  # [B, N, 1, 2d]
        
        # 5. 백엔드 모델(GraphWaveNet)을 통한 예측
        # short_term_history: 최근 단기 데이터
        # hidden_states: STD-MAE에서 추출한 표현
        y_hat = self.backend(short_term_history, hidden_states=hidden_states)
        
        # 6. 출력 형태 변환: [B, N, L_pred] -> [B, L_pred, N, 1]
        y_hat = y_hat.transpose(1, 2).unsqueeze(-1)
        
        return y_hat
