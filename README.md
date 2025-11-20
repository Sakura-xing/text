# 我的模型架构图

这里展示了基于 LDM 的汉服生成模型架构。

```mermaid
graph TD
    %% 定义样式
    classDef texture fill:#e1f5fe,stroke:#01579b,stroke-width:2px;
    classDef pose fill:#fff3e0,stroke:#e65100,stroke-width:2px;
    classDef text fill:#f3e5f5,stroke:#4a148c,stroke-width:2px;
    classDef core fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px;
    classDef lora fill:#ffcdd2,stroke:#b71c1c,stroke-width:2px,stroke-dasharray: 5 5;

    subgraph Inputs ["输入数据层"]
        I_ref(参考图像 I_ref)
        M_ref(服饰 Mask M_ref)
        Pose_in(姿态图 DensePose)
        Txt_in(文本 Prompt)
    end

    subgraph Stream1 ["纹理参考流 (Texture Stream)"]
        direction TB
        VAE_enc[VAE Encoder]
        L_ref[潜在特征 L_ref]
        L_mask[Mask特征 L_mask]
        Concat{Concat & Zero-Conv}
        UNetF[<b>UNet-F</b><br/>(Reference Encoder)<br/>集成 FDA 模块]
        
        I_ref --> VAE_enc --> L_ref
        M_ref --> VAE_enc --> L_mask
        L_ref & L_mask --> Concat --> UNetF
    end
    class Stream1 texture

    subgraph Stream2 ["姿态结构流 (Pose Stream)"]
        CN[<b>ControlNet</b><br/>(Pose Encoder)]
    end
    class Stream2 pose

    subgraph Stream3 ["语义风格流 (Semantic Stream)"]
        CLIP[CLIP Text Encoder]
    end
    class Stream3 text

    subgraph CoreModel ["生成核心 (Generation Core)"]
        UNetG[<b>UNet-G</b><br/>(Denoising UNet)<br/>集成 SRA 模块]
        LoRA((<b>LoRA 权重</b><br/>W_LoRA))
    end
    class UNetG core
    class LoRA lora

    %% 连接关系
    Pose_in --> CN
    CN --F_pose (残差连接)--> UNetG
    
    Txt_in --> CLIP
    CLIP --C_text (Cross-Attn attn2)--> UNetG

    UNetF --"K_ref, V_ref (SRA attn1)"--> UNetG
    
    LoRA -.注入.-> UNetF
    LoRA -.同步注入.-> UNetG

    %% 去噪过程
    Noise(噪声输入 x_t) --> UNetG
    UNetG --> Output(去噪结果 x_t-1)

    %% 样式应用
    linkStyle default stroke:#333,stroke-width:1.5px;
```
