import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'fixed'))
from final_binding import bind as parse_answer

def case(variant='good'):
    citation='D2' if variant=='wrong_citation' else 'D1'
    text='At week 4, the adult training group scored 18 points versus 12 in the control group; this does not establish a week-12 effect.'
    if variant=='wrong_fact':text='At week 4, the adult training group scored 12 points versus 18 in the control group; the effect is guaranteed at week 12.'
    return dict(question='Compare the adult training and control scores at week 4. State whether the evidence establishes a week-12 effect.',
        requirements=[dict(id='Q1',description='Report both adult group scores at week 4.',weight=1),dict(id='Q2',description='Do not infer a week-12 effect from week-4 measurements; state the limitation.',weight=1)],constraints={},
        records=[dict(kind='final',raw_completion=f'<answer>{text} <cite id="{citation}">Report</cite></answer>',source_headers={},evidence=[
            dict(source_id='D1',text='At week 4, adult training participants scored 18 points and adult controls scored 12 points. No week-12 outcomes were collected.'),
            dict(source_id='D2',text='The training centre opened in 2019. It has three classrooms.')])])
