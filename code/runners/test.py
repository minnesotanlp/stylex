import json
import numpy as np

from sklearn.metrics import f1_score

from utils.visualization import VisRecord, visualize_text

class Tester:
    @staticmethod
    def accuracy_test(idx, suffix):
        # Read data
        with open(f'predictions/{idx}_{suffix}.json', 'r') as f:
            data = json.load(f)

        n_correct = 0
        all_preds, all_labels = [], []
        for i in range(len(data)):
            record = data[str(i)]
            style_pred = np.argmax(record['style_prediction'])
            style_label = record['style_label']
            all_preds.append(style_pred)
            all_labels.append(style_label)
            if style_pred == style_label:
                n_correct += 1

        accuracy = n_correct / len(data)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        print(f"Accuracy Test | Accuracy: {accuracy:.5f} | F1 Score: {f1:.5f}")

    @staticmethod
    def correlation_test(idx, suffix):
        # Read data
        with open(f'predictions/{idx}_{suffix}.json', 'r') as f:
            data = json.load(f)

        pred_list, target_list = np.array([]), np.array([])
        for i in range(len(data)):
            record = data[str(i)]
            perception_pred = record['perception_pred']
            perception_label = record['perception_label']
            pred_list = np.concatenate((pred_list, np.array(perception_pred)))
            target_list = np.concatenate((target_list, np.array(perception_label)))

        corr = np.corrcoef(pred_list, target_list)[0, 1]
        print(f"Pearson's Correlation Coefficient Score: {corr:.5f}")

    @staticmethod
    def print_test(filename,
                   prediction_path1,
                   prediction_path2=None,
                   name1=None,
                   name2=None,
                   num_sample=5000,
                   captum=None,
                   ground_truth=True,
                   baseline_path=None,
                   num_changed=5000,
                   split_html=True,
                   ):
        # Read idx1
        with open(prediction_path1, 'r') as f1:
            data1 = json.load(f1)
            
        if name1 is None:
            name1 = prediction_path1

        # Read idx2
        if prediction_path2 is not None:
            with open(prediction_path2, 'r') as f2:
                data2 = json.load(f2)
                
            if name2 is None:
                name2 = prediction_path2
        else:
            data2 = None

        # Read captum
        if captum is not None:
            with open(captum, 'r') as f:
                captum_data = json.load(f)
        else:
            captum_data = None
            
        # Read baseline
        if baseline_path is not None:
            with open(baseline_path, 'r') as f:
                baseline_data = json.load(f)
        else:
            baseline_data = None
        num_corrected, num_worsened = 0, 0

        for i in range(min(num_sample, len(data1))):
            record1 = data1[str(i)]

            if data2 is not None:
                record2 = data2[str(i)]
                assert record1['text'] == record2['text']
            else:
                record2 = None
                
            if baseline_data is not None:
                baseline_record = baseline_data[str(i)]
                assert record1['text'] == baseline_record['text']
            else:
                baseline_record = None

            pred_label1 = np.argmax(record1['style_prediction'])
            style_label = record1['style_label']
            
            if baseline_record is None:
                if split_html:
                    if pred_label1 == record1['style_label']:
                        record_file = filename.replace('.html', '_correct.html')
                    else:
                        record_file = filename.replace('.html', '_wrong.html')
                else:
                    record_file = filename
            else:
                pred_baseline = np.argmax(baseline_record['style_prediction'])
                if pred_label1 == style_label and style_label != pred_baseline and num_corrected < num_changed:
                    record_file = filename.replace('.html', '_corrected.html')
                    num_corrected += 1
                elif pred_label1 != style_label and style_label == pred_baseline and num_worsened < num_changed:
                    record_file = filename.replace('.html', '_worsened.html')
                    num_worsened += 1
                else:
                    continue

            records = [
                VisRecord(record1['idx'], f'{name1}-pred',
                          record1['perception_pred'], np.max(record1['style_prediction']),
                          pred_label1, style_label, record1['text']),
            ]
            if ground_truth:
                records.append(
                    VisRecord(record1['idx'], f'{name1}-GT',
                              record1['perception_label'], np.max(record1['style_prediction']),
                              pred_label1, style_label, record1['text']))
                
            if data2 is not None:
                records.insert(1,
                               VisRecord(record2['idx'], f'{name2}-pred',
                                         record2['perception_pred'], np.max(record2['style_prediction']),
                                         np.argmax(record2['style_prediction']), record2['style_label'],
                                         record2['text']))

            if captum_data is not None:
                captum_record = captum_data[str(i)]
                records.append(
                    VisRecord(captum_record['idx'], 'captum', captum_record['perception_pred'], "", "", "",
                              captum_record['text'])
                )

            html = visualize_text(records)
            with open(record_file, 'a') as f:
                f.write(html.data)
