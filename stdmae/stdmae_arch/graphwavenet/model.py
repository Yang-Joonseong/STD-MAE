import torch
from torch import nn
import torch.nn.functional as F


class nconv(nn.Module):
    """
    그래프 컨볼루션 연산을 수행하는 클래스
    인접 행렬 A와 노드 특성 x 간의 메시지 패싱을 구현
    """
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        """
        그래프 컨볼루션 연산 수행
        Args:
            x (torch.Tensor): 입력 특성 [배치, 채널, 노드, 시간]
            A (torch.Tensor): 인접 행렬 [노드, 노드] 또는 [배치, 노드, 노드]
        Returns:
            torch.Tensor: 그래프 컨볼루션 결과
        """
        A = A.to(x.device)  # 인접 행렬을 x와 같은 디바이스로 이동
        
        # 배치별 인접 행렬인 경우(3차원)와 공유 인접 행렬인 경우(2차원) 처리
        if len(A.shape) == 3:
            # einsum: 'ncvl,nvw->ncwl'은 노드 차원(v)에 대해 메시지 패싱
            # n: 배치, c: 채널, v/w: 노드, l: 시간
            x = torch.einsum('ncvl,nvw->ncwl',(x,A))
        else:
            x = torch.einsum('ncvl,vw->ncwl',(x,A))
        
        return x.contiguous()  # 메모리 연속성 보장


class linear(nn.Module):
    """
    1x1 컨볼루션을 통한 선형 변환 레이어
    채널 차원의 변환을 담당
    """
    def __init__(self,c_in,c_out):
        """
        Args:
            c_in (int): 입력 채널 수
            c_out (int): 출력 채널 수
        """
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self,x):
        """
        Args:
            x (torch.Tensor): 입력 텐서 [배치, 채널, 노드, 시간]
        Returns:
            torch.Tensor: 변환된 출력 [배치, c_out, 노드, 시간]
        """
        return self.mlp(x)


class gcn(nn.Module):
    """
    다중 홉(multi-hop) 그래프 컨볼루션 네트워크 레이어
    여러 인접 행렬과 차수(order)를 지원
    """
    def __init__(self,c_in,c_out,dropout,support_len=3,order=2):
        """
        Args:
            c_in (int): 입력 채널 수
            c_out (int): 출력 채널 수
            dropout (float): 드롭아웃 비율
            support_len (int): 지원하는 인접 행렬 수
            order (int): 그래프 컨볼루션의 차수 (k-hop 이웃까지 고려)
        """
        super(gcn,self).__init__()
        self.nconv = nconv()
        # 입력 채널을 확장: 원본 + (차수 × 인접행렬 수)
        c_in = (order*support_len+1)*c_in
        self.mlp = linear(c_in,c_out)
        self.dropout = dropout
        self.order = order

    def forward(self,x,support):
        """
        Args:
            x (torch.Tensor): 입력 특성 [배치, 채널, 노드, 시간]
            support (list): 인접 행렬 리스트
        Returns:
            torch.Tensor: GCN 출력 [배치, c_out, 노드, 시간]
        """
        out = [x]  # 원본 특성 저장
        
        # 각 인접 행렬에 대해 처리
        for a in support:
            x1 = self.nconv(x,a)  # 1차 이웃 특성
            out.append(x1)
            
            # 2차 이상의 이웃(multi-hop) 처리
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1,a)  # k차 이웃 특성
                out.append(x2)
                x1 = x2  # 다음 차수 계산을 위한 업데이트

        # 모든 결과 연결 (원본 + 모든 홉 결과)
        h = torch.cat(out,dim=1)
        h = self.mlp(h)  # 채널 차원 변환
        h = F.dropout(h, self.dropout, training=self.training)  # 드롭아웃 적용
        return h


class GraphWaveNet(nn.Module):
    """
    GraphWaveNet 모델 - 시공간 그래프 모델링을 위한 딥 러닝 아키텍처
    TCN(Temporal Convolutional Network)과 GCN(Graph Convolutional Network)을 결합
    STD-MAE와 통합되어 시공간 예측 성능 향상

    Paper: Graph WaveNet for Deep Spatial-Temporal Graph Modeling.
    Link: https://arxiv.org/abs/1906.00121
    """

    def __init__(self, num_nodes, supports, dropout=0.3, gcn_bool=True, addaptadj=True, aptinit=None, in_dim=2,out_dim=12,residual_channels=32,dilation_channels=32,skip_channels=256,end_channels=512,kernel_size=2,blocks=4,layers=2, **kwargs):
        """
        Args:
            num_nodes (int): 그래프의 노드 수
            supports (list): 미리 정의된 인접 행렬 리스트
            dropout (float): 드롭아웃 비율
            gcn_bool (bool): GCN 사용 여부
            addaptadj (bool): 적응형 인접 행렬 사용 여부
            aptinit (torch.Tensor): 적응형 인접 행렬 초기화 값
            in_dim (int): 입력 특성 차원
            out_dim (int): 출력(예측) 길이
            residual_channels (int): 잔차 연결 채널 수
            dilation_channels (int): 확장 컨볼루션 채널 수
            skip_channels (int): 스킵 연결 채널 수
            end_channels (int): 최종 컨볼루션 채널 수
            kernel_size (int): 시간 컨볼루션의 커널 크기
            blocks (int): TCN 블록 수
            layers (int): 각 블록 내 레이어 수
        """

        super(GraphWaveNet, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj

        # 모듈 리스트 초기화
        self.filter_convs = nn.ModuleList()  # 필터 컨볼루션
        self.gate_convs = nn.ModuleList()    # 게이트 컨볼루션
        self.residual_convs = nn.ModuleList()  # 잔차 연결 컨볼루션
        self.skip_convs = nn.ModuleList()    # 스킵 연결 컨볼루션
        self.bn = nn.ModuleList()            # 배치 정규화
        self.gconv = nn.ModuleList()         # 그래프 컨볼루션
        self.in_dim=in_dim
        
        # STD-MAE의 표현을 처리하기 위한 MLP
        # Temporal 표현(96차원)을 256차원으로 변환
        self.fc_his_t = nn.Sequential(nn.Linear(96, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU())
        # Spatial 표현(96차원)을 256차원으로 변환
        self.fc_his_s = nn.Sequential(nn.Linear(96, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU())
        
        # 입력 컨볼루션 - 초기 특성 매핑
        self.start_conv = nn.Conv2d(in_channels=in_dim, out_channels=residual_channels, kernel_size=(1,1))
        self.supports = supports

        receptive_field = 1  # 수용 영역 초기화

        # 인접 행렬 수 계산
        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)

        # 적응형 인접 행렬 설정
        if gcn_bool and addaptadj:
            if aptinit is None:
                # 랜덤 초기화
                if supports is None:
                    self.supports = []
                # 학습 가능한 노드 임베딩 (노드 간 관계를 학습)
                self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10), requires_grad=True)
                self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes), requires_grad=True)
                self.supports_len +=1
            else:
                # SVD 기반 초기화
                if supports is None:
                    self.supports = []
                m, p, n = torch.svd(aptinit)
                initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
                initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
                self.nodevec1 = nn.Parameter(initemb1, requires_grad=True)
                self.nodevec2 = nn.Parameter(initemb2, requires_grad=True)
                self.supports_len += 1

        # TCN 블록 구성
        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # 확장 컨볼루션 (dilated convolution) 설정
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                  out_channels=dilation_channels,
                                                  kernel_size=(1,kernel_size),dilation=new_dilation))

                self.gate_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                out_channels=dilation_channels,
                                                kernel_size=(1, kernel_size),dilation=new_dilation))

                # 잔차 연결을 위한 1x1 컨볼루션
                self.residual_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                    out_channels=residual_channels,
                                                    kernel_size=(1, 1)))

                # 스킵 연결을 위한 1x1 컨볼루션
                self.skip_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                out_channels=skip_channels,
                                                kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                
                # 확장 비율 증가 (2의 거듭제곱)
                new_dilation *= 2
                
                # 수용 영역 업데이트
                receptive_field += additional_scope
                additional_scope *= 2
                
                # GCN 추가 (필요시)
                if self.gcn_bool:
                    self.gconv.append(gcn(dilation_channels,residual_channels,dropout,support_len=self.supports_len))

        # 최종 출력 컨볼루션 레이어 (FC Layer #1, FC Layer #2)
        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels, out_channels=end_channels, kernel_size=(1,1), bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=end_channels, out_channels=out_dim, kernel_size=(1,1), bias=True)

        self.receptive_field = receptive_field



    def forward(self, input, hidden_states):
        """
        GraphWaveNet 순전파
        
        Args:
            input (torch.Tensor): 입력 시계열 데이터 [배치, 시간, 노드, 채널]
            hidden_states (torch.Tensor): STD-MAE의 출력 표현 [배치, 노드, 2*임베딩차원]
                                          처음 96차원은 시간적 표현, 나머지 96차원은 공간적 표현
        
        Returns:
            torch.Tensor: 예측 결과 [배치, 노드, 예측길이]
        """

        # 입력 텐서 재구성: [B, L, N, C] -> [B, C, N, L]
        input = input.transpose(1, 3)
        
        # 패딩 적용 (시간 차원)
        input = nn.functional.pad(input,(1,0,0,0))

        # 입력 채널 선택
        input = input[:, :self.in_dim, :, :]
        in_len = input.size(3)
        
        # 수용 영역에 맞게 패딩
        if in_len<self.receptive_field:
            x = nn.functional.pad(input,(self.receptive_field-in_len,0,0,0))
        else:
            x = input
            
        # 초기 컨볼루션
        x = self.start_conv(x)
        skip = 0  # 스킵 연결 초기화

        # 적응형 인접 행렬 계산
        new_supports = None
        if self.gcn_bool and self.addaptadj and self.supports is not None:
            # 노드 임베딩으로부터 인접 행렬 생성
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            new_supports = self.supports + [adp]

        # WaveNet 레이어 순회
        for i in range(self.blocks * self.layers):
            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*

            residual = x  # 잔차 연결을 위한 복사
            
            # 확장 컨볼루션
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)  # tanh 활성화
            
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)  # sigmoid 활성화
            
            # 게이팅 메커니즘 (WaveNet의 핵심)
            x = filter * gate

            # 스킵 연결 처리
            s = x
            s = self.skip_convs[i](s)
            
            # 스킵 연결 차원 맞추기
            try:
                skip = skip[:, :, :,  -s.size(3):]
            except:
                skip = 0
                
            # 스킵 연결 합산
            skip = s + skip


            # GCN 또는 잔차 컨볼루션 적용
            if self.gcn_bool and self.supports is not None:
                if self.addaptadj:
                    x = self.gconv[i](x, new_supports)  # 적응형 인접 행렬 사용
                else:
                    x = self.gconv[i](x,self.supports)  # 고정 인접 행렬 사용
            else:
                x = self.residual_convs[i](x)  # GCN 없이 1x1 컨볼루션만 적용

            # 잔차 연결 합산
            x = x + residual[:, :, :, -x.size(3):]

            # 배치 정규화
            x = self.bn[i](x)

        # STD-MAE의 Temporal 표현 처리
        hidden_states_t = self.fc_his_t(hidden_states[:,:,:96])        # [B, N, 96] -> [B, N, 256]
        hidden_states_t = hidden_states_t.transpose(1, 2).unsqueeze(-1)  # [B, N, 256] -> [B, 256, N, 1]
        skip = skip + hidden_states_t  # 스킵 연결에 Temporal 표현 추가
        
        # STD-MAE의 Spatial 표현 처리
        hidden_states_s = self.fc_his_s(hidden_states[:,:,96:])        # [B, N, 96] -> [B, N, 256]
        hidden_states_s = hidden_states_s.transpose(1, 2).unsqueeze(-1)  # [B, N, 256] -> [B, 256, N, 1]
        skip = skip + hidden_states_s  # 스킵 연결에 Spatial 표현 추가
        
        # 최종 출력 계산 (FC Layer #1, #2)
        x = F.relu(skip)  # ReLU 활성화
        x = F.relu(self.end_conv_1(x))  # [B, 256, N, 1] -> [B, 512, N, 1]
        x = self.end_conv_2(x)  # [B, 512, N, 1] -> [B, 12, N, 1]

        # 출력 텐서 재구성: [B, P, N, 1] -> [B, N, P]
        # P는 예측 길이(out_dim)로, 일반적으로 12시간의 미래 예측값
        x = x.squeeze(-1).transpose(1, 2)
        return x
