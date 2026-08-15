import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def create_charts():
    # Set style
    sns.set_theme(style="whitegrid")
    
    # Data for Recall@1
    models = ['Qwen2.5-1.5B\n(Scalar-Affine)', 'Llama-3.2-3B\n(Full-Rank)']
    untrained_r1 = [3.2, 45.2]
    trained_r1 = [3.2, 24.2]
    
    # Data for Recall@10
    untrained_r10 = [19.4, 59.7]
    trained_r10 = [22.6, 46.8]

    x = np.arange(len(models))
    width = 0.35

    # Chart 1: Recall@1 Comparison
    fig, ax = plt.subplots(figsize=(8, 6))
    
    rects1 = ax.bar(x - width/2, untrained_r1, width, label='Untrained SelfIE', color='#3498db')
    rects2 = ax.bar(x + width/2, trained_r1, width, label='Trained Adapter', color='#2ecc71')

    ax.set_ylabel('Recall@1 (%)', fontsize=14, fontweight='bold')
    ax.set_title('SelfIE Replication: Recall@1 by Model Architecture', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=12)
    ax.legend(fontsize=12)

    # Add text labels on top of the bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=11, fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()
    out_dir = os.path.join(os.path.dirname(__file__), "..", "assets")
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, "recall_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Chart 2: Recall@10 Comparison
    fig, ax = plt.subplots(figsize=(8, 6))
    
    rects1 = ax.bar(x - width/2, untrained_r10, width, label='Untrained SelfIE', color='#2980b9')
    rects2 = ax.bar(x + width/2, trained_r10, width, label='Trained Adapter', color='#27ae60')

    ax.set_ylabel('Recall@10 (%)', fontsize=14, fontweight='bold')
    ax.set_title('SelfIE Replication: Recall@10 by Model Architecture', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=12)
    ax.legend(fontsize=12)

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()
    plt.savefig(os.path.join(out_dir, "recall10_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Charts saved to {out_dir}/")

if __name__ == "__main__":
    create_charts()
