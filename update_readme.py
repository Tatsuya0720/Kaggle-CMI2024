import os
import re
from glob import glob

import nbformat


# CVスコアの取得関数
def get_cv_score(notebook_path, keyword):
    with open(notebook_path) as f:
        notebook = nbformat.read(f, as_version=4)
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                for output in cell.get('outputs', []):
                    if output.output_type == "stream" and keyword in output.text:
                        if "tuned" in keyword:
                            fig_ = f"{float(output.text.split(':')[1].strip()):.3f}"
                        else:
                            last_line = output['text'].strip().splitlines()[-1]
                            fig_ = re.search(r'CV:\s*([\d.]+)', last_line)
                            if fig_:
                                fig_ = fig_.group(1)
                                fig_ = f"{float(fig_):.3f}"
                            else:
                                fig_ = None
                        return fig_
    return None

# 実験結果をREADMEに追記
def update_readme(readme_path, experiments):
    with open(readme_path, 'w') as f:
        f.write("# Experiment Results\n\n")
        f.write("| Experiment | CV | Tuned CV |\n")
        f.write("|------------|----|----------|\n")
        for exp_name, original_cv, tuned_cv in experiments:
            f.write(f"| {exp_name} | {original_cv if original_cv is not None else 'None'} | {tuned_cv if tuned_cv is not None else 'None'} |\n")

# メイン処理
def main():
    base_dir = "."  # 実験フォルダのあるベースディレクトリ
    readme_path = os.path.join(base_dir, "readme.md")
    experiments = []
    experiments_paths = glob("./experiments/**/exp.ipynb", recursive=True)

    # ディレクトリごとにCVスコアを取得
    for experiment in sorted(experiments_paths):
        folder = experiment.split("/")[-2]
        notebook_path = experiment
        try:
            if os.path.isfile(notebook_path):
                cv_score = get_cv_score(notebook_path, keyword="CV:")
                tuned_cv_score = get_cv_score(notebook_path, keyword="tuned Kappa:")

                if cv_score is not None:
                    experiments.append((folder, cv_score, tuned_cv_score))
        except:
           experiments.append((folder, "None", "None"))
    # READMEを更新
    update_readme(readme_path, experiments)

if __name__ == "__main__":
    main()
