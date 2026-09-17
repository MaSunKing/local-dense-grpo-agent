"""Cross-check core judgments; do not let contradictory cores compensate each other."""
from engine import aggregate,indexed,scope_signal,atomic_support_signal,ATOMIC_SUPPORT_FIELDS

def guarded_aggregate(bundle,judgments,scorer,collector_lookup=None,task_modes=None):
    report=aggregate(bundle,judgments,scorer,collector_lookup,task_modes)
    if bundle['kind']!='state':return report
    truth=indexed(judgments['state_truth']['result']['coverage'])
    cites=indexed(judgments.get('state_citation',{}).get('result',{}).get('items',[]))
    updates=indexed(bundle['decoded']['updates']);conflicts=[]
    for k,u in updates.items():
        t=truth[k];c=cites.get(k)
        atomic=bundle['views']['state_truth'].get('decision_contract')=='atomic_component_outcome_population_v2'
        if atomic:
            _,truth_status=atomic_support_signal({name:t[name] for name in ATOMIC_SUPPORT_FIELDS})
            citation_signal,_=atomic_support_signal({name:c[name] for name in ATOMIC_SUPPORT_FIELDS}) if c else (None,'unobservable')
            citation_grade=(None if citation_signal is None else
                            1. if u['status']=='partial' and citation_signal>0 else
                            1. if u['status']=='direct' and citation_signal==1 else
                            .5 if u['status']=='direct' and citation_signal==.5 else 0.)
        else:
            _,truth_status=scope_signal({name:t[name] for name in ('material_help','complete_scope','reason')})
            citation_grade=None if not c else {'correct':1.,'partial':.5,'incorrect':0.,'unobservable':None}[c['verdict']]
        # If truth says this label is fully justified by these chunks, its actual
        # cited superset cannot simultaneously be judged insufficient for that label.
        # Merely equal numbers or equal output labels alone are not a consistency proof.
        if c and u['status']==truth_status and truth_status in {'direct','partial'} and set(t['evidence_ids'])<=set(u['evidence_ids']) and citation_grade is not None and citation_grade<1:
            conflicts.append(k)
    if conflicts:
        report=report|dict(score=None,status='needs_core_review',core_conflicts=conflicts,raw_core_preserved=True)
    return report
