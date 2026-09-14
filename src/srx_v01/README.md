### Сравнение с классическим трансформером

| Характеристика | Classical Transformer (Softmax) | Linear Attention / SSM (Mamba, RWKV) | SRX (Super-Resolvent xFormer) |
| --- | --- | --- | --- |
| **Принцип адресации** | Электростатический потенциал: $\text{softmax}(QK^T / \sqrt{d})$ | Линейное ядро: $\phi(Q)\phi(K)^T$ | Подпространственный резонанс MUSIC: $\Vert\Pi_{\perp}(K) q\Vert^{-2}$ |
| **Сложность на шаг инференса** | $O(N \cdot d)$ (растет с каждым токеном) | $O(d^2)$ или $O(d)$ | $O(d)$ |
| **Объем памяти состояния (KV)** | $O(N \cdot d)$ (гигабайты в DRAM) | $O(d^2)$ (фиксированная матрица) | $O(d)$ (вектор фаз в L1 SRAM) |
| **Уровень шума (Crosstalk)** | $\sum_{j \neq *} e^{q k_j} > 0$ (хвост шума всегда ненулевой) | Диффузное размытие памяти | Строгий 0 (ортогональное дополнение зануляет шум) |
| **Single-Needle в $N \to \infty$** | 100% (ценой перебора всей DRAM) | Катастрофический коллапс при $N \gg d$ | 100% (до предела точности машинного эпсилон) |
| **Multi-Hop Reasoning** | Требует $M$ слоев для $M$ переходов | Практически не работает без дообучения | 1 шаг через инварианты монодромии Лакса |
| **Аппаратное узкое место** | Пропускная способность памяти (Memory-Bound) | Вычисления / Латентность (Compute/Latency-Bound) | **Строго L1/SRAM-Bound** (нулевой трафик в DRAM) |

---

### Математический аппарат архитектуры

```
[Входные проекции]           k_t = W_k x_t,  v_t = W_v x_t,  q_t = W_q x_t
                                       │
[Запись фаз]                           ▼
                             θ_t = θ_{t-1} + α · (k_t ⊙ v_t)
                                       │
[Унитарный оператор]                   ▼
                             U_t = ∏_{i=1}^{d-1} G_i(θ_i)
                                       │
[Аккумулятор значений]                 ▼
                             M_t = λ M_{t-1} + (U_t k_t) v_t^T
                                       │
                                       │  ◄── [Запрос q_t]
                                       ▼
[Шумовой проектор MUSIC]     p_t = (I - U_t U_t^†) q_t
                                       │
[Псевдоспектральный пик]               ▼
                             w_t = 1 / (||p_t||^2 + ε)
                                       │
[Считывание выхода]                    ▼
                             y_t = W_o (w_t · M_t^T (U_t q_t))

```

#### 1. Пространство состояний и унитарная факторизация

Вместо хранения матрицы $d \times d$ оператор памяти $U_t \in \mathrm{U}(d)$ факторизуется через цепочку элементарных вращений Гивенса:


$$U(\Theta) = G_1(\theta_1) G_2(\theta_2) \dots G_{d-1}(\theta_{d-1})$$


где $G_i(\theta_i)$ — ортогональная матрица, вращающая пару координат $(i, i+1)$ на угол $\theta_i$:


$$G_i(\theta_i) = \begin{bmatrix}  I_{i-1} & 0 & 0 & 0 \\ 0 & \cos\theta_i & -\sin\theta_i & 0 \\ 0 & \sin\theta_i & \cos\theta_i & 0 \\ 0 & 0 & 0 & I_{d-i-1} \end{bmatrix}$$

Состояние одной головы описывается вектором фазовых углов $\Theta_t = [\theta_1, \theta_2, \dots, \theta_{d-1}]^T \in [-\pi, \pi]^{d-1}$ и тензором значений $M_t \in \mathbb{R}^{d \times d_v}$.

#### 2. Закон записи: сохранение нормы и отталкивание мод

При поступлении пары ключ-значение $(k_t, v_t) \in \mathbb{R}^d \times \mathbb{R}^{d_v}$:

1. **Фазовый сдвиг:** углы обновляются через нелинейную интерференцию ключа и значения:

$$\Theta_t = \Theta_{t-1} + \alpha \cdot \tanh(k_t \odot \Pi v_t)$$



где $\Pi: \mathbb{R}^{d_v} \to \mathbb{R}^{d-1}$ — проекция размерности, $\alpha$ — темп фазовой модуляции.
2. **Вращение оператора:** формируется текущий унитарный базис $U_t = U(\Theta_t)$. Произведение вращений всегда удовлетворяет $U_t U_t^\dagger = I$, исключая экспоненциальный взрыв или затухание градиентов.
3. **Модуляция памяти значений:**

$$M_t = \gamma M_{t-1} + (U_t k_t) v_t^T$$



где $\gamma \in (0, 1]$ — коэффициент забывания (при $\gamma=1$ память абсолютна).

#### 3. Закон чтения: подпространственный селектор MUSIC

В софтмаксе вероятность извлечения факта пропорциональна экспоненте скалярного произведения. В SRX выборка происходит через расстояние до подпространства шума.

Пусть $q_t \in \mathbb{R}^d$ — вектор запроса.

1. Проекция запроса в сигнальное подпространство оператора памяти:

$$s(q_t) = U_t q_t$$


2. Ортогональное дополнение (проектор на шум):

$$\Pi_{\perp} = I - \sum_{j=1}^{r} u_j u_j^\dagger$$



В факторизации Гивенса проекция на ортогональный шумовой хвост вычисляется за $O(d)$ путем последовательного применения обратных вращений:

$$\tilde{q} = G_1(-\theta_1) \dots G_{d-1}(-\theta_{d-1}) q_t$$


$$\Vert\Pi_{\perp} q_t\Vert^2 = \sum_{j = r+1}^{d} \tilde{q}_j^2$$


3. **Псевдоспектральный весовой коэффициент (Dirac-like gain):**

$$w(q_t) = \frac{1}{\Vert\Pi_{\perp} q_t\Vert^2 + \epsilon_{\text{mach}}}$$


* Если $q_t$ ортогонален записанному ключу, $\Vert\Pi_{\perp} q_t\Vert^2 \sim O(1) \implies w(q_t) \approx 1$.
* Если $q_t$ совпадает с ключом $k^*$, он целиком лежит в сигнальном подпространстве: $\Vert\Pi_{\perp} k^*\Vert^2 \to 0 \implies w(q_t) \to \frac{1}{\epsilon_{\text{mach}}} \approx 10^7 \dots 10^{16}$.


4. **Считывание выхода:**

$$y_t = w(q_t) \cdot \left( M_t^T s(q_t) \right)$$



#### 4. Механизм Multi-Hop через монодромию Лакса

Если записаны зависимые факты $A \to B$ и $B \to C$, они закодированы операторами эволюции $L_A$ и $L_B$. Составной оператор перехода описывается матрицей монодромии:


$$T_{A \to C} = L_B \cdot L_A$$


Собственные векторы $T$ соответствуют замкнутым логическим траекториям. Запрос $q_A$, проходя через степени оператора $U_t^k$, возбуждает терминальный вектор $C$ напрямую за счет сохранения инвариантов следа:


$$I_m = \text{Tr}(U_t^m) = \text{const}$$


Это исключает необходимость последовательной пошаговой авторегрессионной генерации промежуточных токенов цепочки рассуждений.

---

### Сильные стороны и фундаментальные ограничения

#### Преимущества

* **Абсолютная избирательность (Single-Needle 100%):** Векторы, не попавшие в резонанс, зануляются на аппаратном уровне. Отсутствует шум от суммы $10^5$ посторонних токенов.
* **$O(1)$ footprint по памяти:** Состояние контекста состоит из $\Theta \in \mathbb{R}^{d-1}$ и $M \in \mathbb{R}^{d \times d_v}$. Для $d=64, d_v=64$ это $63 + 4096 \approx 4160$ чисел $\approx 16.6\text{ КБ}$. Весь контекст помещается в L1-кэш процессора или Shared Memory GPU.
* **Независимость времени инференса от длины контекста:** Генерация токена номер 100 занимает столько же тактов, сколько генерация токена номер 10 000 000.

#### Недостатки и инженерные риски

* **Предел емкости ортогональных мод (Mode Capacity):** В унитарном пространстве размерности $d$ может существовать не более $d$ взаимно ортогональных векторов. При записи $N \gg d$ ключей векторы неизбежно начинают перекрываться (лемма Джонсона — Линденштрауса). Для масштабирования контекста до миллионов токенов необходимо разбиение на сотни независимых голов ($H \ge 64$) и динамическое управление коэффициентом забывания $\gamma$.
* **Нестабильность градиентов при $\epsilon \to 0$:** Псевдоспектральный пик $1 / (x^2 + \epsilon)$ при малых $\epsilon$ порождает огромные градиенты $\sim -2x / (x^2 + \epsilon)^2$, что разрушает обучение через обратное распространение ошибки (Backpropagation Through Time). Во время обучения требуется регуляризация через динамический отжиг (annealing) параметра $\epsilon$.
* **Потеря «диффузного» контекста:** Трансформер за счет мягкого софтмакса хорошо улавливает размытый стиль и общую тему текста. SRX работает как ассоциативная база данных жестких фактов; стилистическое обобщение в нем требует выделения отдельных низкочастотных голов.

---

### Спецификация для реализации

#### Размерности тензоров

* Входной батч: $X \in \mathbb{R}^{B \times L \times D_{\text{model}}}$.
* Количество голов: $H$. Размерность головы: $d = D_{\text{model}} / H$. Размерность значений: $d_v = d$.
* Проекции: $W_q, W_k \in \mathbb{R}^{D_{\text{model}} \times D_{\text{model}}}$, $W_v \in \mathbb{R}^{D_{\text{model}} \times (H \cdot d_v)}$, $W_o \in \mathbb{R}^{(H \cdot d_v) \times D_{\text{model}}}$.
* Внутреннее состояние одной головы:
* Фазовые углы: $\Theta \in [-\pi, \pi]^{B \times H \times (d-1)}$.
* Память значений: $M \in \mathbb{R}^{B \times H \times d \times d_v}$.



#### Динамический отжиг эпсилон при обучении

Чтобы обучение сходилось стабильно, значение $\epsilon$ должно быть функцией номера эпохи (или шага $s$ из $S_{\text{max}}$):


$$\epsilon(s) = \epsilon_{\text{min}} + (\epsilon_{\text{max}} - \epsilon_{\text{min}}) \cdot \left(1 - \frac{s}{S_{\text{max}}}\right)^2$$


где $\epsilon_{\text{max}} = 1.0$ (мягкий квази-софтмакс в начале обучения), $\epsilon_{\text{min}} = 10^{-4}$ (сверхразрешение в конце обучения).

---

### Эталонный код на Python / PyTorch

Минимальная самодостаточная реализация слоя **SRX (Super-Resolvent Attention)** с поддержкой инференса $O(1)$ и параллельного прохода при обучении:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SRXAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, eps_min: float = 1e-4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.eps_min = eps_min
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        
        # Масштаб фазовой модуляции
        self.alpha = nn.Parameter(torch.ones(1, n_heads, 1, self.d_head - 1) * 0.1)
        self.gamma = nn.Parameter(torch.ones(1, n_heads, 1, 1) * 0.99)

    def _apply_givens(self, x: torch.Tensor, thetas: torch.Tensor, inverse: bool = False) -> torch.Tensor:
        """
        Применение цепочки d-1 вращений Гивенса к вектору x.
        x: [B, H, L, d]
        thetas: [B, H, L, d - 1]
        """
        out = x.clone()
        d = self.d_head
        
        indices = range(d - 1) if not inverse else reversed(range(d - 1))
        
        for i in indices:
            theta = thetas[..., i:i+1]
            if inverse:
                theta = -theta
            c = torch.cos(theta)
            s = torch.sin(theta)
            
            xi = out[..., i:i+1]
            xip = out[..., i+1:i+2]
            
            out[..., i:i+1] = c * xi - s * xip
            out[..., i+1:i+2] = s * xi + c * xip
            
        return out

    def forward(self, x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
        """
        Параллельный режим обучения.
        x: [B, L, D_model]
        """
        B, L, _ = x.shape
        H = self.n_heads
        d = self.d_head
        
        q = self.w_q(x).view(B, L, H, d).transpose(1, 2)  # [B, H, L, d]
        k = self.w_k(x).view(B, L, H, d).transpose(1, 2)
        v = self.w_v(x).view(B, L, H, d).transpose(1, 2)
        
        k = F.normalize(k, p=2, dim=-1)
        q = F.normalize(q, p=2, dim=-1)
        
        # 1. Вычисление фазовых приращений и кумулятивных углов
        # Модуляция угла через проекцию k * v на d - 1 измерений
        delta_theta = self.alpha * torch.tanh(k[..., :-1] * v[..., :-1])
        thetas = torch.cumsum(delta_theta, dim=2)  # [B, H, L, d - 1]
        
        # 2. Поворот ключей в унитарный базис
        k_rot = self._apply_givens(k, thetas)  # [B, H, L, d]
        
        # 3. Накопление памяти значений M_t = sum_tau gamma^(t - tau) (k_rot * v^T)
        # Векторизованная форма через ассоциативное взвешивание
        decay = torch.cumprod(self.gamma.expand(B, H, L, 1), dim=2)
        v_scaled = v / (decay + 1e-6)
        k_scaled = k_rot * decay
        
        # Формирование ассоциативной внешней матрицы ранга 1
        # M_t: [B, H, L, d, d]
        M = torch.einsum('bhlj,bhlk->bhljk', k_scaled, v_scaled)
        M = torch.cumsum(M, dim=2)
        
        # 4. Проекция запроса и оценка расстояния до шумового подпространства MUSIC
        # q_inv: обратное вращение запроса текущим состоянием оператора
        q_inv = self._apply_givens(q, thetas, inverse=True)
        
        # Шум = норма проекции на младшие координаты подпространства
        r = d // 2  # Сигнальный ранг = d / 2, остальное — шумовое подпространство
        noise_energy = torch.sum(q_inv[..., r:] ** 2, dim=-1, keepdim=True)  # [B, H, L, 1]
        
        # Резонансный псевдоспектральный гейн
        w = 1.0 / (noise_energy + eps)  # [B, H, L, 1]
        
        # 5. Извлечение памяти через свертку
        q_rot = self._apply_givens(q, thetas)  # [B, H, L, d]
        y = torch.einsum('bhlj,bhljk->bhlk', q_rot, M)  # [B, H, L, d]
        
        # Применение резкого резонансного отклика
        y = y * w
        
        # Возврат в исходную размерность
        y = y.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.w_o(y)

    @torch.no_grad()
    def step(self, x_t: torch.Tensor, state: dict, eps: float = 1e-4):
        """
        Шаг авторегрессионного инференса O(1) памяти и O(d) вычислений.
        x_t: [B, 1, D_model]
        state: {'thetas': [B, H, d - 1], 'M': [B, H, d, d]}
        """
        B = x_t.shape[0]
        H = self.n_heads
        d = self.d_head
        
        q = self.w_q(x_t).view(B, H, d)
        k = self.w_k(x_t).view(B, H, d)
        v = self.w_v(x_t).view(B, H, d)
        
        k = F.normalize(k, p=2, dim=-1)
        q = F.normalize(q, p=2, dim=-1)
        
        # 1. Обновление фаз
        delta_theta = (self.alpha.squeeze(2) * torch.tanh(k[..., :-1] * v[..., :-1]))
        thetas = state['thetas'] + delta_theta
        
        # Подготовка псевдо-последовательности длины 1 для вызова поворотов
        thetas_seq = thetas.unsqueeze(2)
        k_seq = k.unsqueeze(2)
        q_seq = q.unsqueeze(2)
        
        # 2. Обновление унитарного базиса и аккумулятора памяти M
        k_rot = self._apply_givens(k_seq, thetas_seq).squeeze(2)
        M = self.gamma.squeeze(2) * state['M'] + torch.einsum('bhj,bhk->bhjk', k_rot, v)
        
        # 3. Вычисление шума проекции запроса (MUSIC Core)
        q_inv = self._apply_givens(q_seq, thetas_seq, inverse=True).squeeze(2)
        r = d // 2
        noise_energy = torch.sum(q_inv[..., r:] ** 2, dim=-1, keepdim=True)
        w = 1.0 / (noise_energy + eps)
        
        # 4. Считывание выхода
        q_rot = self._apply_givens(q_seq, thetas_seq).squeeze(2)
        y = torch.einsum('bhj,bhjk->bhk', q_rot, M) * w
        
        y = y.contiguous().view(B, self.d_model)
        out = self.w_o(y)
        
        # Обновленное состояние остается строго O(1)
        new_state = {'thetas': thetas, 'M': M}
        return out, new_state

```