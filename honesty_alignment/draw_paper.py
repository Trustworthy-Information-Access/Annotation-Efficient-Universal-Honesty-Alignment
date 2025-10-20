import pandas as pd
import numpy as np
import orjson
import os
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import math
from matplotlib.ticker import FixedLocator
from prepare_label import calculate_consistency_conf_per_n_elements_parallel
import scipy.stats as stats
from scipy.stats import spearmanr

def read_json(path):
    qa_data = []
    with open(path, 'rb') as f:  # 二进制模式，orjson需要bytes
        for line in f:
            qa_data.append(orjson.loads(line))
    return qa_data


def read_xlsx(file_path):
    # 读取 Excel 文件
    df = pd.read_excel(file_path, header=None)  # 默认读取第一个sheet
    prob_score = df.iloc[:, 0]
    consis_score = df.iloc[:, 1]
    greedy_score = df.iloc[:, 2]
    sft_score = df.iloc[:, 3]
    hybrid_score = df.iloc[:, 4]
    return prob_score, consis_score, greedy_score, sft_score, hybrid_score


def get_weighted_avg_scores(domain='in_domain', qa_type='short', base_path='', model_name='Qwen2.5-7B-Instruct', plot_type='auroc'):
    in_domain_datasets = ['nq', 'tq', 'hq', '2wikimultihopqa', 'pararel_patterns']
    ood_datasets = ['complex_web_questions', 'musique', 'web_questions', 'popqa', 'squad']
    used_datasets = in_domain_datasets if domain == 'in_domain' else ood_datasets

    prob_all = []
    consis_all = []
    greedy_all = []
    sft_all = []
    hybrid_all = []
    data_cnt_all = []
    for dataset in used_datasets:
        if model_name == 'Qwen2.5-7B-Instruct' and plot_type == 'auroc':
            file_path = f'{base_path}/{dataset}/{qa_type}_alignment_{plot_type}.xlsx'
            if not os.path.exists(file_path):
                file_path = f'{base_path}/{dataset}/{qa_type}_alignment.xlsx'
        else:
            file_path = f'{base_path}/{dataset}/{qa_type}_alignment_{plot_type}.xlsx'
        prob_score, consis_score, greedy_score, sft_score, hybrid_score = read_xlsx(file_path) # 按照列读取, 从左到右分别是5种分数, 每一行对应一个标注数据量
        qa_data = read_json(f'/data/users/nishiyu/res/perception_training/res/{model_name}/{dataset}/test_data/{qa_type}_qa/{dataset}_test_{model_name}_{qa_type}_qa_0.0_0.95_50_sample_1.jsonl')
        data_cnt = len(qa_data)
        data_cnt_all.append(data_cnt)

        prob_all.append(prob_score.values)
        consis_all.append(consis_score.values)
        greedy_all.append(greedy_score.values)
        sft_all.append(sft_score.values)
        hybrid_all.append(hybrid_score.values)

    weights = np.array(data_cnt_all) / sum(data_cnt_all) # 按数据量加权
    prob_avg = np.round(np.average(prob_all, axis=0, weights=weights), 2)
    consis_avg = np.round(np.average(consis_all, axis=0, weights=weights), 2)
    greedy_avg = np.round(np.average(greedy_all, axis=0, weights=weights), 2)
    sft_avg = np.round(np.average(sft_all, axis=0, weights=weights), 2)
    hybrid_avg = np.round(np.average(hybrid_all, axis=0, weights=weights), 2)
    print(f'\n==== {qa_type.upper()} QA {domain.upper()} 平均得分 ====')
    print(f'Prob score avg:   {prob_avg}')
    print(f'Consis score avg: {consis_avg}')
    print(f'Greedy score avg: {greedy_avg}')
    print(f'SFT score avg:    {sft_avg}')
    print(f'Hybrid score avg: {hybrid_avg}')
    return prob_avg, consis_avg, greedy_avg, sft_avg, hybrid_avg

def get_scores_for_each_dataset(qa_type='short', base_path='', dataset='', model_name='Qwen2.5-7B-Instruct', plot_type='auroc'):
    file_path = f'{base_path}/{dataset}/{qa_type}_alignment_{plot_type}.xlsx'
    prob_score, consis_score, greedy_score, sft_score, hybrid_score = read_xlsx(file_path)

    prob_avg = [round(item,2) for item in prob_score.values]
    consis_avg = [round(item,2) for item in consis_score.values]
    greedy_avg = [round(item,2) for item in greedy_score.values]
    sft_avg = [round(item,2) for item in sft_score.values]
    hybrid_avg = [round(item,2) for item in hybrid_score.values]
    print(f'\n==== {dataset} 平均得分 ====')
    print(f'Prob score avg:   {prob_avg}')
    print(f'Consis score avg: {consis_avg}')
    print(f'Greedy score avg: {greedy_avg}')
    print(f'SFT score avg:    {sft_avg}')
    print(f'Hybrid score avg: {hybrid_avg}')
    return prob_avg, consis_avg, greedy_avg, sft_avg, hybrid_avg

def plot_long_qa_baselines():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    # 数据
    long_avg = []
    names = ['Prob', 'N-Prob', 'Verbal-0', 'Verbal-10', 'Consis-Lex', 'Consis-Sem']

    for model_name in ['Qwen2.5-7B-Instruct', 'Qwen2.5-14B-Instruct', 'Meta-Llama-3-8B-Instruct']:
        file_path = f'/data/users/nishiyu/code/Easy_LLaMA-Factory/perception_training/baseline_res/{model_name}_long_qa_results.xlsx'
        df = pd.read_excel(file_path)
        long_avg.append(df['Back Avg'])

    # 转成 DataFrame 再平均
    long_avg_df = pd.concat(long_avg, axis=1)
    long_avg_mean = long_avg_df.mean(axis=1).round(2).tolist()  # 转 list

    # 设置图形样式
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 8))

    # x轴位置
    n_methods = len(names)
    bar_width = 0.8  # 柱子宽度加大
    x_positions = range(n_methods)

    # 颜色
    colors = ['#E74C3C'] * n_methods  # 只绘制long-QA，红色

    # 绘制柱状图
    bars = ax.bar(x_positions, long_avg_mean, color=colors, alpha=0.8, 
                  edgecolor='white', linewidth=1.2, width=bar_width)

    # 添加数值标签
    for bar, score in zip(bars, long_avg_mean):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{score:.1f}',
                ha='center', va='bottom', fontsize=20, 
                fontweight='bold', color='#2C3E50')

    # 设置x轴标签
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(names, fontsize=22, fontweight='600')

    # 设置y轴标签和范围
    ax.set_ylabel('AUROC', fontsize=22, fontweight='bold', color='#2C3E50')
    ax.set_xlabel('Methods', fontsize=22, fontweight='bold', color='#2C3E50')
    ax.set_ylim(60, 75)

    # 美化网格
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis='y', labelsize=20)

    # 设置背景色
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#FAFAFA')  # 保留背景色

    # 添加标题
    # ax.set_title('Long-QA Baselines Performance', fontsize=24, fontweight='bold', color='#E74C3C', pad=20)

    # 调整布局
    plt.tight_layout()

    # 显示图像
    plt.show()

    # 保存高质量图片
    plt.savefig('../plot_res/Meta-Llama-3-8B-Instruct/long_qa_baselines_performance.pdf', 
                dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')




def plot_in_domain_and_mmlu(plot_type='align', outdir="comparison", outfile="in_domain_mmlu"):
    """
    绘制 1x4 图：
    前三个子图：三个模型在 In-Domain Long-QA 下的曲线
    第四个子图：Qwen2.5-7B-Instruct 在 MMLU OOD Long-QA 下的曲线
    """
    # ==== Step 1: 数据收集 ====
    actual_train_sizes = [1, 2, 4, 6, 8, 10, 20, 30, 50, 80, 200, 580]
    
    # 模型映射
    model_name_map = {
        'Meta-Llama-3-8B-Instruct': 'Llama3-8B',
        'Qwen2.5-7B-Instruct': 'Qwen2.5-7B',
        'Qwen2.5-14B-Instruct': 'Qwen2.5-14B'
    }
    y_label='Alignment' if plot_type == 'align' else 'AUROC'

    # 存放数据
    data_for_all = { "in_domain_long": {}, "mmlu_long": {} }

    # --------- In-domain Long-QA 数据 ---------
    for model_name in ['Qwen2.5-7B-Instruct', 'Qwen2.5-14B-Instruct', 'Meta-Llama-3-8B-Instruct']:
        total_sft, total_hybrid = [], []
        for greedy_training_samples in [0]:
            if greedy_training_samples == 0:
                greedy_tail_name = ''
            elif greedy_training_samples <= 10000:
                k = int(greedy_training_samples / 1000)
                greedy_tail_name = f'/_{k}k_training_samples'
            else:
                k = int(greedy_training_samples / 1000)
                greedy_tail_name = f'/_{k}k_training_samples'
            base_path = f"./res/{model_name}/pararel_patterns-nq-tq-hq-2wikimultihopqa_no_shuffle/{greedy_tail_name}"
            # 使用已有函数取数据
            prob_avg, consis_avg, greedy_avg, sft_avg, hybrid_avg = get_weighted_avg_scores(
                "in_domain", "long", base_path, model_name, plot_type
            )
            total_sft.append(sft_avg)
            total_hybrid.append(hybrid_avg)
        data_for_all["in_domain_long"][model_name] = {"etc": total_hybrid[0], "dlfc": total_sft[0]}

    # --------- MMLU OOD Long-QA 数据 ---------
    model_name = 'Qwen2.5-7B-Instruct'
    total_sft, total_hybrid = [], []
    for greedy_training_samples in [0]:
        if greedy_training_samples == 0:
            greedy_tail_name = ''
        elif greedy_training_samples <= 10000:
            k = int(greedy_training_samples / 1000)
            greedy_tail_name = f'/_{k}k_training_samples'
        else:
            k = int(greedy_training_samples / 1000)
            greedy_tail_name = f'/_{k}k_training_samples'
        base_path = f"./res/{model_name}/pararel_patterns-nq-tq-hq-2wikimultihopqa_no_shuffle/{greedy_tail_name}"
        prob_avg, consis_avg, greedy_avg, sft_avg, hybrid_avg = get_scores_for_each_dataset(
            "long", base_path, "mmlu", model_name, plot_type
        )
        total_sft.append(sft_avg)
        total_hybrid.append(hybrid_avg)
    data_for_all["mmlu_long"][model_name] = {"etc": total_hybrid[0], "dlfc": total_sft[0]}

    # ==== Step 2: 绘制 ====
    plt.rcParams.update({
        'font.size': 22,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
        'axes.linewidth': 1.0,
        'axes.labelsize': 23,
        'axes.titlesize': 24,
        'axes.titleweight': 'bold',
        'xtick.labelsize': 21,
        'ytick.labelsize': 21,
        'legend.fontsize': 25,
        'figure.dpi': 300,
        'savefig.dpi': 600,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.3,
    })

    # 颜色和样式
    etc_style = dict(linestyle='-', marker='o', markersize=6, linewidth=2.2,
                     markerfacecolor='white', markeredgewidth=2, alpha=0.9,
                     color='#2E86AB', markeredgecolor='#2E86AB', zorder=3)
    dlfc_style = dict(linestyle='--', marker='s', markersize=5, linewidth=2.2,
                      markerfacecolor='white', markeredgewidth=2, alpha=0.9,
                      color='#E63946', markeredgecolor='#E63946', zorder=3)

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    fig.patch.set_facecolor('#f8f9fa')

    # 前三张：in-domain long
    for i, model_name in enumerate(['Qwen2.5-7B-Instruct', 'Qwen2.5-14B-Instruct', 'Meta-Llama-3-8B-Instruct']):
        ax = axes[i]
        display_name = model_name_map.get(model_name, model_name) + ' (In-Domain)'
        etc_data = data_for_all["in_domain_long"][model_name]["etc"]
        dlfc_data = data_for_all["in_domain_long"][model_name]["dlfc"]
        ax.plot(actual_train_sizes, etc_data, label="ETC", **etc_style)
        ax.plot(actual_train_sizes, dlfc_data, label="DLFC", **dlfc_style)
        ax.set_xscale('log', base=2)
        ax.set_xticks([2**i for i in range(10)])
        ax.set_xticklabels([f'$2^{{{i}}}$' for i in range(10)])
        ax.set_xlim(0.8, 1024)
        if plot_type == 'auroc':
            ax.set_ylim(70, 90)
        else:
            ax.set_ylim(70, 80) # for align
        ax.set_title(display_name, fontweight='bold')
        ax.set_xlabel("Annotation Data Size (k)")
        if i == 0:
            ax.set_ylabel(y_label)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_facecolor('#f8f9fa')
        ax.grid(True, linestyle='-', alpha=0.6, color='white', linewidth=3, which='major', zorder=1)
        ax.grid(True, linestyle='-', alpha=0.3, color='white', linewidth=1, which='minor', zorder=1)
        
        # 启用次刻度
        ax.minorticks_on()
        
        # 设置刻度线样式
        ax.tick_params(axis='x', which='major', 
                        direction='inout', length=8, width=1.2, 
                        color='#666666', labelcolor='#333333')
        ax.tick_params(axis='x', which='minor', 
                        direction='inout', length=4, width=0.8, 
                        color='#999999')
        ax.tick_params(axis='y', which='major', 
                        direction='inout', length=8, width=1.2, 
                        color='#666666', labelcolor='#333333')
        ax.tick_params(axis='y', which='minor', 
                        direction='inout', length=4, width=0.8, 
                        color='#999999')

    # 第四张：MMLU long
    ax = axes[3]
    etc_data = data_for_all["mmlu_long"]['Qwen2.5-7B-Instruct']["etc"]
    dlfc_data = data_for_all["mmlu_long"]['Qwen2.5-7B-Instruct']["dlfc"]
    ax.plot(actual_train_sizes, etc_data, label="ETC", **etc_style)
    ax.plot(actual_train_sizes, dlfc_data, label="DLFC", **dlfc_style)
    ax.set_xscale('log', base=2)
    ax.set_xticks([2**i for i in range(10)])
    ax.set_xticklabels([f'$2^{{{i}}}$' for i in range(10)])
    ax.set_xlim(0.8, 1024)
    if plot_type == 'auroc':
        ax.set_ylim(60, 80)
    else:
        ax.set_ylim(60, 75) # for align
    ax.set_title("Qwen2.5-7B (MMLU)", fontweight='bold')
    ax.set_xlabel("Annotation Data Size (k)")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_facecolor('#f8f9fa')
    ax.grid(True, linestyle='-', alpha=0.6, color='white', linewidth=3, which='major', zorder=1)
    ax.grid(True, linestyle='-', alpha=0.3, color='white', linewidth=1, which='minor', zorder=1)
    
    # 启用次刻度
    ax.minorticks_on()
    
    # 设置刻度线样式
    ax.tick_params(axis='x', which='major', 
                    direction='inout', length=8, width=1.2, 
                    color='#666666', labelcolor='#333333')
    ax.tick_params(axis='x', which='minor', 
                    direction='inout', length=4, width=0.8, 
                    color='#999999')
    ax.tick_params(axis='y', which='major', 
                    direction='inout', length=8, width=1.2, 
                    color='#666666', labelcolor='#333333')
    ax.tick_params(axis='y', which='minor', 
                    direction='inout', length=4, width=0.8, 
                    color='#999999')

    # 图例
    handles = [
        plt.Line2D([0], [0], **{**etc_style, 'label': 'EliCal'}),
        plt.Line2D([0], [0], **{**dlfc_style, 'label': 'Cal-Only'})
    ]
    fig.legend(handles, ['EliCal', 'Cal-Only'],
               bbox_to_anchor=(0.5, 1.02), loc='center', ncol=2,
               frameon=True, framealpha=0.98, facecolor='#ffffff',
               edgecolor='#dddddd', fontsize=16)

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)

    # 保存
    if not os.path.exists(f'../plot_res/{outdir}'):
        os.makedirs(f'../plot_res/{outdir}')
    base_filename = f'../plot_res/{outdir}/{outfile}'
    plt.savefig(f'{base_filename}.pdf', dpi=600, bbox_inches='tight',
                facecolor='#ffffff', edgecolor='none', pad_inches=0.2)
    plt.show()
    print(f"Saved to {base_filename}.pdf")

def plot_ood_only(plot_type='align', outdir="comparison", outfile="ood_only"):
    """
    绘制 1x3 图：
    三个子图：三个模型在 In-Domain Long-QA 下的曲线
    """
    # ==== Step 1: 数据收集 ====
    actual_train_sizes = [1, 2, 4, 6, 8, 10, 20, 30, 50, 80, 200, 580]
    
    # 模型映射
    model_name_map = {
        'Meta-Llama-3-8B-Instruct': 'Llama3-8B',
        'Qwen2.5-7B-Instruct': 'Qwen2.5-7B',
        'Qwen2.5-14B-Instruct': 'Qwen2.5-14B'
    }
    y_label='Alignment' if plot_type == 'align' else 'AUROC'

    # 存放数据
    data_for_all = { "in_domain_long": {} }

    # --------- In-domain Long-QA 数据 ---------
    for model_name in ['Qwen2.5-7B-Instruct', 'Qwen2.5-14B-Instruct', 'Meta-Llama-3-8B-Instruct']:
        total_sft, total_hybrid = [], []
        for greedy_training_samples in [0]:
            if greedy_training_samples == 0:
                greedy_tail_name = ''
            elif greedy_training_samples <= 10000:
                k = int(greedy_training_samples / 1000)
                greedy_tail_name = f'/_{k}k_training_samples'
            else:
                k = int(greedy_training_samples / 1000)
                greedy_tail_name = f'/_{k}k_training_samples'
            base_path = f"./res/{model_name}/pararel_patterns-nq-tq-hq-2wikimultihopqa_no_shuffle/{greedy_tail_name}"
            prob_avg, consis_avg, greedy_avg, sft_avg, hybrid_avg = get_weighted_avg_scores(
                "ood", "long", base_path, model_name, plot_type
            )
            total_sft.append(sft_avg)
            total_hybrid.append(hybrid_avg)
        data_for_all["in_domain_long"][model_name] = {"etc": total_hybrid[0], "dlfc": total_sft[0]}

    # ==== Step 2: 绘制 ====
    plt.rcParams.update({
        'font.size': 22,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
        'axes.linewidth': 1.0,
        'axes.labelsize': 23,
        'axes.titlesize': 24,
        'axes.titleweight': 'bold',
        'xtick.labelsize': 21,
        'ytick.labelsize': 21,
        'legend.fontsize': 25,
        'figure.dpi': 300,
        'savefig.dpi': 600,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.3,
    })

    # 颜色和样式
    etc_style = dict(linestyle='-', marker='o', markersize=6, linewidth=2.2,
                     markerfacecolor='white', markeredgewidth=2, alpha=0.9,
                     color='#2E86AB', markeredgecolor='#2E86AB', zorder=3)
    dlfc_style = dict(linestyle='--', marker='s', markersize=5, linewidth=2.2,
                      markerfacecolor='white', markeredgewidth=2, alpha=0.9,
                      color='#E63946', markeredgecolor='#E63946', zorder=3)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))  # 改成1x3
    fig.patch.set_facecolor('#f8f9fa')

    # 三张 in-domain 子图
    for i, model_name in enumerate(['Qwen2.5-7B-Instruct', 'Qwen2.5-14B-Instruct', 'Meta-Llama-3-8B-Instruct']):
        ax = axes[i]
        display_name = model_name_map.get(model_name, model_name) + ' (OOD)'
        etc_data = data_for_all["in_domain_long"][model_name]["etc"]
        dlfc_data = data_for_all["in_domain_long"][model_name]["dlfc"]
        ax.plot(actual_train_sizes, etc_data, label="ETC", **etc_style)
        ax.plot(actual_train_sizes, dlfc_data, label="DLFC", **dlfc_style)
        ax.set_xscale('log', base=2)
        ax.set_xticks([2**i for i in range(10)])
        ax.set_xticklabels([f'$2^{{{i}}}$' for i in range(10)])
        ax.set_xlim(0.8, 1024)
        # if plot_type == 'auroc':
        ax.set_ylim(70, 90)
        # else:
        # ax.set_ylim(70, 80)  # for align
        ax.set_title(display_name, fontweight='bold')
        ax.set_xlabel("Annotation Data Size (k)")
        if i == 0:
            ax.set_ylabel(y_label)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_facecolor('#f8f9fa')
        ax.grid(True, linestyle='-', alpha=0.6, color='white', linewidth=3, which='major', zorder=1)
        ax.grid(True, linestyle='-', alpha=0.3, color='white', linewidth=1, which='minor', zorder=1)
        ax.minorticks_on()
        ax.tick_params(axis='x', which='major', direction='inout', length=8, width=1.2, color='#666666', labelcolor='#333333')
        ax.tick_params(axis='x', which='minor', direction='inout', length=4, width=0.8, color='#999999')
        ax.tick_params(axis='y', which='major', direction='inout', length=8, width=1.2, color='#666666', labelcolor='#333333')
        ax.tick_params(axis='y', which='minor', direction='inout', length=4, width=0.8, color='#999999')

    # 图例
    handles = [
        plt.Line2D([0], [0], **{**etc_style, 'label': 'EliCal'}),
        plt.Line2D([0], [0], **{**dlfc_style, 'label': 'Cal-Only'})
    ]
    fig.legend(handles, ['EliCal', 'Cal-Only'],
               bbox_to_anchor=(0.5, 1.02), loc='center', ncol=2,
               frameon=True, framealpha=0.98, facecolor='#ffffff',
               edgecolor='#dddddd', fontsize=16)

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)

    # 保存
    if not os.path.exists(f'../plot_res/{outdir}'):
        os.makedirs(f'../plot_res/{outdir}')
    base_filename = f'../plot_res/{outdir}/{outfile}'
    plt.savefig(f'{base_filename}.pdf', dpi=600, bbox_inches='tight',
                facecolor='#ffffff', edgecolor='none', pad_inches=0.2)
    plt.show()
    print(f"Saved to {base_filename}.pdf")


def plot_effects_of_training_size_for_elicitation():
    def plot_for_all_dataset_across_greedy_training_samples(qa_type, dataset, prob_avg, consis_avg, greedy_avg, sft_avg, hybrid_avg, outdir):
        """
        绘制elicitation效果随预训练数据量的变化曲线，精美版本
        横轴使用真实数值（线性尺度），标注每个点相对于consis_avg[0][0]的百分比
        """
        import matplotlib.pyplot as plt
        import numpy as np
        from scipy.optimize import curve_fit
        from scipy.interpolate import interp1d
        import math
        import os
        
        def logarithmic_func(x, a, b, c):
            return a * np.log(x + b) + c

        pretrain_data_k = [10, 20, 30, 50, 80, 200, 580]  
        pretrain_labels = ['10K', '20K', '30K', '50K', '80K', '200K', '580K']

        # ---------- 精美风格设置 ----------
        plt.style.use('default')  # 重置样式
        plt.rcParams.update({
            'font.family': ['Times New Roman', 'DejaVu Serif', 'serif'],
            'font.size': 20,
            'axes.linewidth': 1.5,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'grid.linewidth': 0.8,
            'figure.facecolor': 'white',
            'axes.facecolor': '#fafafa',
            'text.color': '#2C3E50',
            'axes.labelcolor': '#2C3E50',
            'xtick.color': '#2C3E50',
            'ytick.color': '#2C3E50'
        })

        # 颜色方案
        primary_color = '#6C5CE7'  # 优雅紫色
        secondary_color = '#74B9FF'  # 柔和蓝色
        accent_color = '#FD79A8'  # 温暖粉色
        text_color = '#2C3E50'  # 深灰蓝

        # 创建图形
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        # ============ 数据处理 ============
        greedy_values = [greedy_avg[i][0] for i in range(len(pretrain_labels))]
        baseline_value = consis_avg[0][0]  # 基准值
        percentages = [(val / baseline_value) * 100 for val in greedy_values]
        # ============ 绘制基准线 ============
        # ============ 绘制基准线 ============
        baseline_value = consis_avg[0][0]
        ax.axhline(y=baseline_value, color=secondary_color, linestyle='--', linewidth=2,
                alpha=0.8, zorder=0, label='Consis-Sem')
        
        # 添加基准线标签
        ax.text(max(pretrain_data_k) * 0.98, baseline_value * 1.01, 
                f'Consis-Sem: {baseline_value:.3f}',
                color=secondary_color, fontsize=20, fontweight='bold',
                va='bottom', ha='right', alpha=0.8)

        # ============ 绘制平滑曲线 ============
        x_smooth = np.linspace(min(pretrain_data_k), max(pretrain_data_k), 500)
        
        try:
            popt, _ = curve_fit(logarithmic_func, pretrain_data_k, greedy_values, maxfev=20000)
            y_smooth = logarithmic_func(x_smooth, *popt)
            # 绘制渐变效果的曲线
            ax.plot(x_smooth, y_smooth, color=primary_color, linewidth=4, 
                    alpha=0.8, zorder=2, label='Elicitation Performance')
            # 添加阴影效果
            ax.fill_between(x_smooth, y_smooth, alpha=0.15, color=primary_color, zorder=1)
        except Exception:
            f_interp = interp1d(pretrain_data_k, greedy_values, kind='cubic', fill_value='extrapolate')
            y_smooth = f_interp(x_smooth)
            ax.plot(x_smooth, y_smooth, color=primary_color, linewidth=4, 
                    alpha=0.8, zorder=2, label='Elicitation Performance')
            ax.fill_between(x_smooth, y_smooth, alpha=0.15, color=primary_color, zorder=1)

        # ============ 绘制数据点 ============
        scatter = ax.scatter(pretrain_data_k, greedy_values, 
                            color=accent_color, s=200, zorder=5,
                            marker='o', edgecolors='white', linewidth=3, 
                            alpha=0.95, label='Data Points')

        # 添加内圈
        ax.scatter(pretrain_data_k, greedy_values, 
                color=primary_color, s=80, zorder=6,
                marker='o', alpha=0.8)

    # ============ 添加百分比标注 ============
        placed_positions = []  # 存放已有标注位置，避免重叠
        x_offset_factor = 0.15  # 控制文本相对横坐标的偏移比例（可调）

        min_sep = 0.05 * (max(greedy_values) - min(greedy_values))  # 最小垂直间隔

        for i, (x, y, pct) in enumerate(zip(pretrain_data_k, greedy_values, percentages)):
            # 初始位置：先放在点上方
            y_pos = y + min_sep
            x_offset = x_offset_factor * x  

            # 避免和已有标注重叠
            while any(abs(y_pos - yy) < min_sep for yy in placed_positions):
                y_pos += min_sep  

            placed_positions.append(y_pos)

            # ax.annotate(f'{pct:.1f}%', 
            #         xy=(x, y), xytext=(x + x_offset, y_pos),
            #         ha='center', va='bottom',
            #         fontsize=16, fontweight='bold',
            #         rotation=-30,  # 🔥 旋转 30 度
            #         rotation_mode='anchor',  # 保证旋转时锚点对齐
            #         color=text_color,
            #         bbox=dict(boxstyle='round,pad=0.3',
            #                   facecolor='white',
            #                   edgecolor=primary_color,
            #                   alpha=0.9),
            #         arrowprops=dict(arrowstyle='->',
            #                         color=primary_color,
            #                         alpha=0.7,
            #                         lw=1.5),
            #         zorder=7)


        # ============ 坐标轴设置 ============
        # 智能刻度
        max_pow = int(math.ceil(math.log2(max(pretrain_data_k))))
        min_pow = int(math.floor(math.log2(min(pretrain_data_k))))
        candidate_ticks = [2**i for i in range(min_pow, max_pow + 1)]
        
        if len(candidate_ticks) > 6:
            step = math.ceil(len(candidate_ticks)/6)
            ticks = candidate_ticks[::step]
        else:
            ticks = candidate_ticks
        
        tick_labels = [f'$2^{{{int(np.log2(t))}}}$' for t in ticks]

        ax.set_xticks(ticks)
        ax.set_xticklabels(tick_labels, fontsize=20, color=text_color)
        
        # 设置坐标轴范围
        x_pad = (max(pretrain_data_k) - min(pretrain_data_k)) * 0.08
        y_pad = (max(greedy_values) - min(greedy_values)) * 0.1
        ax.set_xlim(min(pretrain_data_k)-x_pad, max(pretrain_data_k)+x_pad)
        ax.set_ylim(min(greedy_values)-y_pad, 75)

        # ============ 标签和标题 ============
        ax.set_xlabel('Data Size for Confidence Elicitation (k)', 
                    fontsize=20, fontweight='bold', color=text_color)
        ax.set_ylabel('AUROC', 
                    fontsize=20, fontweight='bold', color=text_color)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)


        # ============ 网格优化 ============
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.8)
        ax.set_axisbelow(True)

        # ============ 保存图片 ============
        plt.tight_layout(pad=2.0)
        
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        filename = f'{qa_type}_{dataset}_elicitation_scaling_beautiful.pdf'
        filepath = os.path.join(outdir, filename)
        plt.savefig(filepath, dpi=400, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
        plt.show()

        print(f"精美图片已保存到: {filepath}")
        print(f"基准值 (consis_avg[0][0]): {baseline_value:.4f}")
        print("各数据点相对基准值的百分比:")
        for label, pct in zip(pretrain_labels, percentages):
            print(f"  {label}: {pct:.1f}%")

    plot_type = 'auroc'
    data_for_all_settings = {}
    for domain in ['in_domain']:
        for qa_type in ['long']:
            data_across_models = {}
            for model_name in ['Qwen2.5-7B-Instruct']:
                print(f'{domain}-{qa_type}-{model_name}')
                total_prob = []
                total_consis = []
                total_greedy = []
                total_sft = []
                total_hybrid = []
                for greedy_training_samples in [10000, 20000, 30000, 50000, 80000, 200000, 0]:
                    if greedy_training_samples == 0:
                        greedy_epochs=10
                        greedy_tail_name = ''
                    elif greedy_training_samples <= 10000:
                        greedy_epochs=50
                        k=int(greedy_training_samples/1000)
                        greedy_tail_name=f'/_{k}k_training_samples'
                    else:
                        greedy_epochs=15
                        k=int(greedy_training_samples/1000)
                        greedy_tail_name=f'/_{k}k_training_samples'
                    base_path = f"./res/{model_name}/pararel_patterns-nq-tq-hq-2wikimultihopqa_no_shuffle/{greedy_tail_name}"
                    outdir = f'{model_name}/pararel_patterns-nq-tq-hq-2wikimultihopqa_no_shuffle_greedy_training_samples/{greedy_training_samples}'
                    # 一种预训练数据量下, 不同方法随标注数据量变化的AUROC分数
                    prob_avg, consis_avg, greedy_avg, sft_avg, hybrid_avg = get_weighted_avg_scores(domain, qa_type, base_path, model_name, plot_type)
                    # prob_avg, consis_avg, greedy_avg, sft_avg, hybrid_avg = get_scores_for_each_dataset(qa_type, base_path, 'mmlu', model_name)
                    total_prob.append(prob_avg)
                    total_consis.append(consis_avg)
                    total_greedy.append(greedy_avg)
                    total_sft.append(sft_avg)
                    total_hybrid.append(hybrid_avg)
                    data_across_models[model_name] = {'etc': total_hybrid, 'dlfc': total_sft}
                plot_for_all_dataset_across_greedy_training_samples(qa_type, domain, total_prob, total_consis, total_greedy, total_sft, total_hybrid, outdir)


def plot_in_and_ood_mlp(plot_type='auroc', outdir="comparison", outfile="in_and_ood"):
    """
    绘制 2x3 图：
    第一行：三个模型在 In-Domain Long-QA 下的曲线
    第二行：三个模型在 OOD Long-QA 下的曲线
    每列代表一个模型
    """
    import os
    import matplotlib.pyplot as plt

    # ==== Step 1: 数据收集 ====
    actual_train_sizes = [1, 2, 4, 6, 8, 10, 20, 30, 50, 80, 200, 580]
    
    model_name_map = {
        'Meta-Llama-3-8B-Instruct': 'Llama3-8B',
        'Qwen2.5-7B-Instruct': 'Qwen2.5-7B',
        'Qwen2.5-14B-Instruct': 'Qwen2.5-14B'
    }
    y_label = 'Alignment' if plot_type == 'align' else 'AUROC'

    data_for_all = { "in_domain_long": {}, "ood_long": {} }

    # --------- In-domain & OOD Long-QA 数据 ---------
    for model_name in ['Qwen2.5-7B-Instruct', 'Qwen2.5-14B-Instruct', 'Meta-Llama-3-8B-Instruct']:
        # In-domain
        prob_avg, consis_avg, greedy_avg, sft_avg, hybrid_avg = get_weighted_avg_scores(
            "in_domain", "long",
            f"./res/{model_name}/pararel_patterns-nq-tq-hq-2wikimultihopqa_no_shuffle/mlp",
            model_name, plot_type
        )
        data_for_all["in_domain_long"][model_name] = {"etc": hybrid_avg, "dlfc": sft_avg}

        # OOD
        prob_avg, consis_avg, greedy_avg, sft_avg, hybrid_avg = get_weighted_avg_scores(
            "ood", "long",
            f"./res/{model_name}/pararel_patterns-nq-tq-hq-2wikimultihopqa_no_shuffle/mlp",
            model_name, plot_type
        )
        data_for_all["ood_long"][model_name] = {"etc": hybrid_avg, "dlfc": sft_avg}

    # ==== Step 2: 绘制 ====
    plt.rcParams.update({
        'font.size': 22,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
        'axes.linewidth': 1.0,
        'axes.labelsize': 23,
        'axes.titlesize': 24,
        'axes.titleweight': 'bold',
        'xtick.labelsize': 21,
        'ytick.labelsize': 21,
        'legend.fontsize': 25,
        'figure.dpi': 300,
        'savefig.dpi': 600,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.3,
    })

    etc_style = dict(linestyle='-', marker='o', markersize=6, linewidth=2.2,
                     markerfacecolor='white', markeredgewidth=2, alpha=0.9,
                     color='#2E86AB', markeredgecolor='#2E86AB', zorder=3)
    dlfc_style = dict(linestyle='--', marker='s', markersize=5, linewidth=2.2,
                      markerfacecolor='white', markeredgewidth=2, alpha=0.9,
                      color='#E63946', markeredgecolor='#E63946', zorder=3)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))  # 2x3
    fig.patch.set_facecolor('#f8f9fa')

    for row, domain in enumerate(["in_domain_long", "ood_long"]):
        for col, model_name in enumerate(['Qwen2.5-7B-Instruct', 'Qwen2.5-14B-Instruct', 'Meta-Llama-3-8B-Instruct']):
            ax = axes[row, col]
            display_name = model_name_map.get(model_name, model_name)
            display_name += " (In-Domain)" if row == 0 else " (OOD)"

            etc_data = data_for_all[domain][model_name]["etc"]
            dlfc_data = data_for_all[domain][model_name]["dlfc"]
            ax.plot(actual_train_sizes, etc_data, label="ETC", **etc_style)
            ax.plot(actual_train_sizes, dlfc_data, label="DLFC", **dlfc_style)
            
            ax.set_xscale('log', base=2)
            ax.set_xticks([2**i for i in range(10)])
            ax.set_xticklabels([f'$2^{{{i}}}$' for i in range(10)])
            ax.set_xlim(0.8, 1024)
            ax.set_ylim(60, 85)
            ax.set_title(display_name, fontweight='bold')
            ax.set_xlabel("Annotation Data Size (k)")
            if col == 0:
                ax.set_ylabel(y_label)

            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_facecolor('#f8f9fa')
            ax.grid(True, linestyle='-', alpha=0.6, color='white', linewidth=3, which='major', zorder=1)
            ax.grid(True, linestyle='-', alpha=0.3, color='white', linewidth=1, which='minor', zorder=1)
            ax.minorticks_on()
            ax.tick_params(axis='x', which='major', direction='inout', length=8, width=1.2, color='#666666', labelcolor='#333333')
            ax.tick_params(axis='x', which='minor', direction='inout', length=4, width=0.8, color='#999999')
            ax.tick_params(axis='y', which='major', direction='inout', length=8, width=1.2, color='#666666', labelcolor='#333333')
            ax.tick_params(axis='y', which='minor', direction='inout', length=4, width=0.8, color='#999999')

    handles = [
        plt.Line2D([0], [0], **{**etc_style, 'label': 'EliCal'}),
        plt.Line2D([0], [0], **{**dlfc_style, 'label': 'Cali-Only'})
    ]
    fig.legend(handles, ['EliCal', 'Cal-Only'],
               bbox_to_anchor=(0.5, 1.02), loc='center', ncol=2,
               frameon=True, framealpha=0.98, facecolor='#ffffff',
               edgecolor='#dddddd', fontsize=16)

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)

    if not os.path.exists(f'../plot_res/{outdir}'):
        os.makedirs(f'../plot_res/{outdir}')
    base_filename = f'../plot_res/{outdir}/{outfile}'
    plt.savefig(f'{base_filename}.pdf', dpi=600, bbox_inches='tight',
                facecolor='#ffffff', edgecolor='none', pad_inches=0.2)
    plt.show()
    print(f"Saved to {base_filename}.pdf")

def plot_in_and_ood_ece(plot_type='ece', outdir="comparison", outfile="in_and_ood"):
    """
    绘制 2x3 图：
    第一行：三个模型在 In-Domain Long-QA 下的曲线
    第二行：三个模型在 OOD Long-QA 下的曲线
    每列代表一个模型
    """
    import os
    import matplotlib.pyplot as plt

    # ==== Step 1: 数据收集 ====
    actual_train_sizes = [1, 2, 4, 6, 8, 10, 20, 30, 50, 80, 200, 580]
    
    model_name_map = {
        'Meta-Llama-3-8B-Instruct': 'Llama3-8B',
        'Qwen2.5-7B-Instruct': 'Qwen2.5-7B',
        'Qwen2.5-14B-Instruct': 'Qwen2.5-14B'
    }
    y_label = 'ECE'

    data_for_all = { "in_domain_long": {}, "ood_long": {} }

    # --------- In-domain & OOD Long-QA 数据 ---------
    for model_name in ['Qwen2.5-7B-Instruct', 'Qwen2.5-14B-Instruct', 'Meta-Llama-3-8B-Instruct']:
        # In-domain
        prob_avg, consis_avg, greedy_avg, sft_avg, hybrid_avg = get_weighted_avg_scores(
            "in_domain", "long",
            f"./res/{model_name}/pararel_patterns-nq-tq-hq-2wikimultihopqa_no_shuffle",
            model_name, plot_type
        )
        data_for_all["in_domain_long"][model_name] = {"etc": hybrid_avg, "dlfc": sft_avg}

        # OOD
        prob_avg, consis_avg, greedy_avg, sft_avg, hybrid_avg = get_weighted_avg_scores(
            "ood", "long",
            f"./res/{model_name}/pararel_patterns-nq-tq-hq-2wikimultihopqa_no_shuffle",
            model_name, plot_type
        )
        data_for_all["ood_long"][model_name] = {"etc": hybrid_avg, "dlfc": sft_avg}

    # ==== Step 2: 绘制 ====
    plt.rcParams.update({
        'font.size': 22,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
        'axes.linewidth': 1.0,
        'axes.labelsize': 23,
        'axes.titlesize': 24,
        'axes.titleweight': 'bold',
        'xtick.labelsize': 21,
        'ytick.labelsize': 21,
        'legend.fontsize': 25,
        'figure.dpi': 300,
        'savefig.dpi': 600,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.3,
    })

    etc_style = dict(linestyle='-', marker='o', markersize=6, linewidth=2.2,
                     markerfacecolor='white', markeredgewidth=2, alpha=0.9,
                     color='#2E86AB', markeredgecolor='#2E86AB', zorder=3)
    dlfc_style = dict(linestyle='--', marker='s', markersize=5, linewidth=2.2,
                      markerfacecolor='white', markeredgewidth=2, alpha=0.9,
                      color='#E63946', markeredgecolor='#E63946', zorder=3)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))  # 2x3
    fig.patch.set_facecolor('#f8f9fa')

    for row, domain in enumerate(["in_domain_long", "ood_long"]):
        for col, model_name in enumerate(['Qwen2.5-7B-Instruct', 'Qwen2.5-14B-Instruct', 'Meta-Llama-3-8B-Instruct']):
            ax = axes[row, col]
            display_name = model_name_map.get(model_name, model_name)
            display_name += " (In-Domain)" if row == 0 else " (OOD)"

            etc_data = data_for_all[domain][model_name]["etc"]
            dlfc_data = data_for_all[domain][model_name]["dlfc"]
            ax.plot(actual_train_sizes, etc_data, label="ETC", **etc_style)
            ax.plot(actual_train_sizes, dlfc_data, label="DLFC", **dlfc_style)
            
            ax.set_xscale('log', base=2)
            ax.set_xticks([2**i for i in range(10)])
            ax.set_xticklabels([f'$2^{{{i}}}$' for i in range(10)])
            ax.set_xlim(0.8, 1024)
            ax.set_ylim(0, 0.3)
            ax.set_title(display_name, fontweight='bold')
            ax.set_xlabel("Annotation Data Size (k)")
            if col == 0:
                ax.set_ylabel(y_label)

            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_facecolor('#f8f9fa')
            ax.grid(True, linestyle='-', alpha=0.6, color='white', linewidth=3, which='major', zorder=1)
            ax.grid(True, linestyle='-', alpha=0.3, color='white', linewidth=1, which='minor', zorder=1)
            ax.minorticks_on()
            ax.tick_params(axis='x', which='major', direction='inout', length=8, width=1.2, color='#666666', labelcolor='#333333')
            ax.tick_params(axis='x', which='minor', direction='inout', length=4, width=0.8, color='#999999')
            ax.tick_params(axis='y', which='major', direction='inout', length=8, width=1.2, color='#666666', labelcolor='#333333')
            ax.tick_params(axis='y', which='minor', direction='inout', length=4, width=0.8, color='#999999')

    handles = [
        plt.Line2D([0], [0], **{**etc_style, 'label': 'EliCal'}),
        plt.Line2D([0], [0], **{**dlfc_style, 'label': 'Cal-Only'})
    ]
    fig.legend(handles, ['EliCal', 'Cal-Only'],
               bbox_to_anchor=(0.5, 1.02), loc='center', ncol=2,
               frameon=True, framealpha=0.98, facecolor='#ffffff',
               edgecolor='#dddddd', fontsize=16)

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)

    if not os.path.exists(f'../plot_res/{outdir}'):
        os.makedirs(f'../plot_res/{outdir}')
    base_filename = f'../plot_res/{outdir}/{outfile}'
    plt.savefig(f'{base_filename}.pdf', dpi=600, bbox_inches='tight',
                facecolor='#ffffff', edgecolor='none', pad_inches=0.2)
    plt.show()
    print(f"Saved to {base_filename}.pdf")

def plot_confidence_values(confidence1, confidence2, model_name, dataset, qa_type, n_bins=1000):
    """
    Plot confidence comparison (line + error bars) and confidence histogram (sample counts) side by side.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import spearmanr

    conf1 = np.array(confidence1)
    conf2 = np.array(confidence2)
    spearman_corr, p_value = spearmanr(conf1, conf2)
    
    # Colors
    colors = {
        'conf1': '#0F4C75',      
        'conf2': '#BB2649',      
        'conf1_light': '#3282B8',
        'conf2_light': '#E84A5F',
        'grid': '#E9ECEF',       
        'background': '#FDFDFD'
    }

    # --- Step 1: Prepare line plot (confidence comparison) ---
    idx_sorted = np.argsort(conf1)
    bin_size = len(conf1) // n_bins
    conf1_binned, conf2_binned, conf1_se, conf2_se = [], [], [], []

    for i in range(n_bins):
        start = i * bin_size
        end = start + bin_size if i < n_bins-1 else len(conf1)
        bin1 = conf1[idx_sorted[start:end]]
        bin2 = conf2[idx_sorted[start:end]]
        conf1_binned.append(np.mean(bin1))
        conf2_binned.append(np.mean(bin2))
        conf1_se.append(np.std(bin1)/np.sqrt(len(bin1)))
        conf2_se.append(np.std(bin2)/np.sqrt(len(bin2)))

    x_bins = conf1_binned

    # --- Step 2: Prepare histogram (sample counts) ---
    hist_bins = np.linspace(0, 1, 11)  # 10 bins: 0.0-0.1, 0.1-0.2, ..., 0.9-1.0
    counts1, _ = np.histogram(conf1, bins=hist_bins)
    counts2, _ = np.histogram(conf2, bins=hist_bins)
    bin_centers = (hist_bins[:-1] + hist_bins[1:]) / 2

    # --- Step 3: Create 1x2 subplot ---
    fig, axes = plt.subplots(1, 2, figsize=(14,6))
    axes[0].set_facecolor(colors['background'])
    axes[1].set_facecolor(colors['background'])

    # --- Left: confidence comparison (sampled with error band between lines) ---
    idx_sorted = np.argsort(conf1)
    n_samples = len(conf1)
    n_bins = 20  # 控制点数量
    bin_edges = np.linspace(0, n_samples, n_bins+1, dtype=int)

    x_vals, conf1_sampled, conf2_sampled = [], [], []

    for start, end in zip(bin_edges[:-1], bin_edges[1:]):
        bin1 = conf1[idx_sorted[start:end]]
        bin2 = conf2[idx_sorted[start:end]]
        x_vals.append((start + end) / 2)
        conf1_sampled.append(np.mean(bin1))
        conf2_sampled.append(np.mean(bin2))

    # 绘制线条
    axes[0].plot(x_vals, conf1_sampled, 'o-', label='Accuracy',
                linewidth=4, markersize=10, color=colors['conf1'],
                markerfacecolor=colors['conf1_light'], markeredgecolor=colors['conf1'])
    axes[0].plot(x_vals, conf2_sampled, 's-', label='Confidence',
                linewidth=4, markersize=10, color=colors['conf2'],
                markerfacecolor=colors['conf2_light'], markeredgecolor=colors['conf2'])

    # 绘制两条线之间的浅阴影
    axes[0].fill_between(x_vals, conf1_sampled, conf2_sampled, color='gray', alpha=0.2)

    axes[0].set_xlabel('Questions Sorted by Accuracy', fontsize=20, fontweight='600')
    axes[0].set_ylabel('Value', fontsize=20, fontweight='600')
    axes[0].set_title(f'Self-Consistency Confidence vs. Accuracy', fontsize=20, fontweight='bold', pad=20)
    axes[0].grid(True, alpha=0.3, color=colors['grid'], linestyle='-', linewidth=0.8)
    axes[0].legend(loc='upper left', frameon=True, fancybox=True, shadow=True,
                framealpha=0.95, fontsize=20, facecolor='#FFFFFF', edgecolor='#BDC3C7')

    # Spearman correlation
    spearman_corr, p_value = spearmanr(conf1, conf2)
    significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
    axes[0].text(0.98, 0.02, f'Spearman ρ = {spearman_corr:.3f}', 
                transform=axes[0].transAxes, fontsize=20,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='#BDC3C7'))
    axes[0].tick_params(axis='both', labelsize=20)



    # --- Right: histogram (sample counts as percentage) ---
    total_samples = len(conf1)
    counts1_pct = counts1 / total_samples * 100
    counts2_pct = counts2 / total_samples * 100

    width = 0.04  # bar width
    # --- Right: histogram (sample counts as percentage) ---
    axes[1].bar(bin_centers - width/2, counts1_pct, width=width, label='Accuracy', color=colors['conf1_light'])
    axes[1].bar(bin_centers + width/2, counts2_pct, width=width, label='Confidence', color=colors['conf2_light'])

    # 设置固定刻度 0.0, 0.1, ..., 1.0
    x_ticks = np.arange(0, 1.01, 0.1)
    axes[1].set_xticks(x_ticks)
    axes[1].set_xticklabels([f"{x:.1f}" for x in x_ticks], fontsize=20)

    axes[1].set_xlabel('Confidence/Accuracy', fontsize=20, fontweight='600')
    axes[1].set_ylabel('Percentage (%)', fontsize=20, fontweight='600')
    axes[1].set_title('Confidence/Accuracy Distribution', fontsize=20, fontweight='bold', pad=20)
    axes[1].grid(True, alpha=0.3, color=colors['grid'], linestyle='-', linewidth=0.8)
    axes[1].legend(frameon=True, fancybox=True, shadow=True, framealpha=0.95, fontsize=20, facecolor='#FFFFFF', edgecolor='#BDC3C7')

    # 坐标轴刻度字体统一改为20
    axes[1].tick_params(axis='y', labelsize=20)

    plt.tight_layout()
    plt.savefig(f'./paper_detail_res/{dataset}_{model_name}_{qa_type}_confidence_line_hist.pdf', 
                dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.show()



def correlation_between_consis_and_acc(model_name='Qwen2.5-7B-Instruct', dataset='tq', qa_type='long_qa'):
    base_dir = '/data/users/nishiyu/code/Easy_LLaMA-Factory/perception_training/baseline_res/record_per_dataset'
    file_path = f'{base_dir}/{model_name}/{qa_type}/{dataset}.jsonl'
    data = read_json(file_path)
    print(f'dataset: {dataset}')
    sorry = []
    sample_acc = []
    acc = []
    sem_conf = []
    
    for item in data:
        sample_acc.append(item['sample_acc'])
        acc.append(item['acc'])
        sem_conf.append(item['consis_sem_conf'])

    corr, p_value = spearmanr(sem_conf, sample_acc)

    print("Spearman correlation:", corr)
    print("p-value:", p_value)
    plot_confidence_values(sample_acc, sem_conf, model_name, dataset, qa_type, 20)
    # plot_ranking_positions(sample_acc, sem_conf, acc, model_name, dataset, qa_type)


if __name__ == '__main__':
    # baselines
    # plot_long_qa_baselines()

    # in-domain and mmlu
    # plot_in_domain_and_mmlu('align')

    # ood
    plot_ood_only('align')

    # plot_in_and_ood_mlp()

    # plot_in_and_ood_ece()

    # correlation_between_consis_and_acc()
