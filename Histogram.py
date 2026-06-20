import os
import glob
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np


def count_dota_objects(base_path):
    """Подсчет объектов по классам из TXT файлов аннотаций DOTA"""

    class_counts = {'train': defaultdict(int), 'val': defaultdict(int)}
    objects_per_image = {'train': [], 'val': []}

    train_path = os.path.join(base_path, 'train txt')
    val_path = os.path.join(base_path, 'val txt')

    if not os.path.exists(train_path) or not os.path.exists(val_path):
        print("Ошибка: папки train txt или val txt не найдены")
        return None, None

    train_txts = glob.glob(os.path.join(train_path, '*.txt'))
    val_txts = glob.glob(os.path.join(val_path, '*.txt'))

    print(f"Найдено файлов: train - {len(train_txts)}, val - {len(val_txts)}")

    for txt_file in train_txts:
        file_counts = process_file(txt_file, class_counts['train'])
        objects_per_image['train'].append(file_counts)

    for txt_file in val_txts:
        file_counts = process_file(txt_file, class_counts['val'])
        objects_per_image['val'].append(file_counts)

    print_results(class_counts)
    plot_histograms(objects_per_image)

    return class_counts, objects_per_image


def process_file(file_path, counter):
    """Обработка одного файла аннотаций"""
    file_counts = defaultdict(int)

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()

                if len(parts) >= 9:
                    class_name = parts[8]
                    counter[class_name] += 1
                    file_counts[class_name] += 1
                elif len(parts) >= 5:
                    for part in parts:
                        try:
                            float(part)
                        except ValueError:
                            if part not in ['difficult', 'easy', 'medium', 'hard']:
                                counter[part] += 1
                                file_counts[part] += 1
                                break
    except:
        pass

    return dict(file_counts)


def print_results(class_counts):
    """Вывод результатов подсчета"""

    total_train = sum(class_counts['train'].values())
    total_val = sum(class_counts['val'].values())
    total_all = total_train + total_val

    print(f"\nОбщее количество объектов: {total_all:,}")
    print(f"Train: {total_train:,}, Val: {total_val:,}")

    target_classes = ['small-vehicle', 'large-vehicle', 'ship']

    print("\nЦелевые классы:")
    for cls in target_classes:
        train_count = class_counts['train'].get(cls, 0)
        val_count = class_counts['val'].get(cls, 0)
        total = train_count + val_count
        proportion = (total / total_all * 100) if total_all > 0 else 0

        print(f"{cls}: train={train_count:,}, val={val_count:,}, всего={total:,} ({proportion:.1f}%)")

    other_train = sum(v for k, v in class_counts['train'].items() if k not in target_classes)
    other_val = sum(v for k, v in class_counts['val'].items() if k not in target_classes)
    other_total = other_train + other_val
    other_proportion = (other_total / total_all * 100) if total_all > 0 else 0

    print(f"остальные: train={other_train:,}, val={other_val:,}, всего={other_total:,} ({other_proportion:.1f}%)")


def plot_histograms(objects_per_image):
    """Построение 6 гистограмм с обрезанным диапазоном для наглядности"""

    target_classes = ['small-vehicle', 'large-vehicle', 'ship']
    colors = {'train': '#3498db', 'val': '#e74c3c'}

    # Для каждого класса считаем разумный максимум оси X (95-й перцентиль)
    # чтобы выбросы не растягивали масштаб
    x_lims = {}
    for cls in target_classes:
        train_data = [fc.get(cls, 0) for fc in objects_per_image['train']]
        val_data = [fc.get(cls, 0) for fc in objects_per_image['val']]
        all_data = train_data + val_data
        # Берем 97-й перцентиль как верхнюю границу отображения
        x_lims[cls] = int(np.percentile(all_data, 97)) + 1

    # Фигура 3x2
    fig, axes = plt.subplots(3, 2, figsize=(14, 14))
    fig.suptitle('Распределение количества объектов по изображениям (train vs val)',
                 fontsize=16, fontweight='bold')

    for row_idx, cls in enumerate(target_classes):
        train_data = [fc.get(cls, 0) for fc in objects_per_image['train']]
        val_data = [fc.get(cls, 0) for fc in objects_per_image['val']]

        x_max = x_lims[cls]
        bins = np.arange(0, x_max + 1, 1)

        for col_idx, (split, data, color) in enumerate([
            ('train', train_data, colors['train']),
            ('val', val_data, colors['val'])
        ]):
            ax = axes[row_idx, col_idx]

            # Обрезаем данные для гистограммы (но считаем статистику по всем)
            clipped_data = [x if x < x_max else x_max for x in data]

            ax.hist(clipped_data, bins=bins, color=color,
                    alpha=0.75, edgecolor='black', linewidth=0.5)

            mean_val = np.mean(data)
            median_val = np.median(data)
            non_zero = sum(1 for x in data if x > 0)
            total_objs = sum(data)
            max_val = max(data)

            # Линии среднего и медианы (рисуем, только если они в пределах оси)
            if mean_val < x_max:
                ax.axvline(mean_val, color='red', linestyle='--',
                           linewidth=1.5, label=f'Среднее: {mean_val:.2f}')
            if median_val < x_max:
                ax.axvline(median_val, color='green', linestyle='-.',
                           linewidth=1.5, label=f'Медиана: {median_val:.1f}')

            stats = (f'Изображений: {len(data)}\n'
                     f'С объектами: {non_zero}\n'
                     f'Всего объектов: {total_objs}\n'
                     f'Мин: {min(data)}, Макс: {max_val}')

            ax.text(0.98, 0.98, stats, transform=ax.transAxes,
                    fontsize=9, verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))

            ax.set_xlim(0, x_max)
            ax.set_xlabel('Количество объектов', fontsize=10)
            ax.set_ylabel('Количество изображений', fontsize=10)
            ax.set_title(f'{cls} — {split.capitalize()}', fontsize=12, fontweight='bold')
            ax.legend(loc='upper left', fontsize=9)
            ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('histograms_train_val_by_class.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Статистика в консоль
    print("\nСтатистика распределения по изображениям:")
    for cls in target_classes:
        train_data = [fc.get(cls, 0) for fc in objects_per_image['train']]
        val_data = [fc.get(cls, 0) for fc in objects_per_image['val']]

        print(f"\n{cls}:")
        print(f"  Train — среднее: {np.mean(train_data):.2f}, "
              f"медиана: {np.median(train_data):.1f}, "
              f"макс: {max(train_data)}, 97-й перцентиль: {int(np.percentile(train_data, 97))}")
        print(f"  Val   — среднее: {np.mean(val_data):.2f}, "
              f"медиана: {np.median(val_data):.1f}, "
              f"макс: {max(val_data)}, 97-й перцентиль: {int(np.percentile(val_data, 97))}")


# Запуск
base_path = r'J:\dataset DOTA\TXT'
counts, objects_per_image = count_dota_objects(base_path)