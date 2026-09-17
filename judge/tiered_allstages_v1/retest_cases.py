"""Explicit partial-support case, separate from the disputed pooled-only fixture."""
from fixtures import pack

def state_case(good=True):
    q='For the laboratory pump test, report measured flow rate AND noise level, and identify the tested operating voltage.'
    req=[dict(id='Q1',description='Report laboratory pump flow rate AND noise level.',weight=1),dict(id='Q2',description='Identify the laboratory test operating voltage.',weight=1)]
    ev=[dict(source_id='FLOW',text='At an operating voltage of 12 V, the laboratory pump delivered 8 litres per minute. The laboratory did not measure noise level.')]
    items=[dict(id='P1',description=req[0]['description']),dict(id='P2',description=req[1]['description'])]
    updates=[dict(id='P1',status='partial' if good else 'direct',evidence_ids=['FLOW']),dict(id='P2',status='direct',evidence_ids=['FLOW'])]
    return pack('state',q,req,ev,dict(updates=updates),dict(items=items))
