"""Held-out transfer domain; no matching terms or expected scores in prompts."""
def case(variant='good'):
    text='The archive retains deleted project files for 45 days. Recovery is available only to workspace administrators; retention does not guarantee successful recovery.'
    if variant=='wrong_fact':text='The archive retains deleted project files for 90 days. Every workspace viewer can recover them, and recovery is guaranteed.'
    if variant=='missing':text='The archive retains deleted project files for 45 days.'
    cite='S2' if variant=='wrong_citation' else 'S1'
    return dict(question='Under the supplied archive policy, how long are deleted project files retained, who can recover them, and is recovery guaranteed?',requirements=[
        dict(id='R1',description='State the retention period for deleted project files.',weight=1),
        dict(id='R2',description='Identify who is permitted to recover deleted project files.',weight=1),
        dict(id='R3',description='State whether retention guarantees successful recovery.',weight=1)],constraints={},records=[dict(kind='final',raw_completion=f'<answer>{text} <cite id="{cite}">Policy</cite></answer>',source_headers={},evidence=[
        dict(source_id='S1',text='Deleted project files are retained for 45 days. Only workspace administrators may recover deleted files. Retention does not guarantee successful recovery.'),
        dict(source_id='S2',text='The service offers light and dark display themes. The dashboard supports keyboard navigation.')])])
