import torch
from captum.attr import visualization
from IPython.core.display import HTML

# Visualization utils
def format_token(token):
    if token in ['[CLS]', '[SEP]']:
        new_token = ''
    elif token.startswith('##'):
        new_token = token.replace('##', '')
    else:
        new_token = f' {token}'
    
    return new_token

def format_word_importances(words, importances):
    if importances is None or len(importances) == 0:
        return "<td></td>"
    assert len(words) <= len(importances)
    tags = ["<td>"]
    for word, importance in zip(words, importances[: len(words)]):
        word = format_token(word)
        if word == '':
            continue
        color = visualization._get_color(importance)
        unwrapped_tag = '<mark style="background-color: {color}; opacity:1.0; \
                    line-height:1.75"><font color="black">{word}</font></mark>'.format(
            color=color, word=word
        )
        tags.append(unwrapped_tag)
    tags.append("</td>")
    return "".join(tags)

def visualize_score(filename, idx, text, style_pred, label, perceptions, names, separate=False):
    assert len(perceptions) == len(names)

    label_pred = torch.argmax(style_pred)
    if separate:
        if label_pred == label:
            filename = filename.replace('.html', '_correct.html')
        else:
            filename = filename.replace('.html', '_wrong.html')

    records = []
    for perception, name in zip(perceptions, names):
        record = VisRecord(
            idx,
            name,
            perception,
            torch.max(style_pred),
            label_pred,
            label,
            text
        )
        records.append(record)

    viz_html = visualize_text(records)
    with open(filename, 'a') as f:
        f.write(viz_html.data)


class VisRecord:
    __slots__ = [
        'idx',
        'type',
        'word_attributions',
        'pred_prob',
        'pred_class',
        'true_class',
        'raw_input',
    ]

    def __init__(self,
                 idx,
                 type,
                 word_attributions,
                 pred_prob,
                 pred_class,
                 true_class,
                 raw_input):
        self.idx = idx
        self.type = type
        self.word_attributions = word_attributions
        self.pred_prob = pred_prob
        self.pred_class = pred_class
        self.true_class = true_class
        self.raw_input = raw_input


def visualize_text(visrecords):
    format_classname = visualization.format_classname
    dom = ['<table width: 100%>']
    rows = [
        '<tr><th>Idx</th>'
        '<th>Type</th>'
        '<th>Predicted Label</th>'
        '<th>True Label</th>'
        '<th>Word Importance</th>'
    ]

    for visrecord in visrecords:
        pred_prob = f"({visrecord.pred_prob:.2f})" if isinstance(visrecord.pred_prob, float) else visrecord.pred_prob
        rows.append(
            ''.join(
                [
                    '<tr>',
                    format_classname(visrecord.idx),
                    format_classname(visrecord.type),
                    format_classname(
                        f'{visrecord.pred_class} {pred_prob}'
                    ),
                    format_classname(visrecord.true_class),
                    format_word_importances(visrecord.raw_input,
                                            visrecord.word_attributions),
                    '<tr>',
                ]
            )
        )
    dom.append(''.join(rows))
    dom.append('</table>')
    html = HTML(''.join(dom))
    return html
