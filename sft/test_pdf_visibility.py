import unittest
from shared_interface import *

class PdfVisibility(unittest.TestCase):
 def test_routes_and_scope(self):
  for kind,level,route,abstract in [('dual_pdf_s2','pmc_full_text','open_pdf_mineru',False),('dual_pdf_abstract_fallback','pubmed_abstract','abstract_fallback',True)]:
   docs=[dict(source_id='S2:a#c0',text='Original passage.',content_level=level),dict(source_id='WEB:b#c0',text='Web passage.')]
   aug=dict(kind=kind,synthetic=True,real_execution=False,id_mapping={'PMID:1':'S2:a'})
   out=pdf_scenario_evidence(docs,aug)
   self.assertEqual(out[0]['reader_route'],route);self.assertEqual(out[0]['abstract_only'],abstract)
   self.assertEqual(out[0]['text'],docs[0]['text']);self.assertEqual(out[1],docs[1]);self.assertNotIn('reader_route',docs[0])
   packed=compact_source_headers({'opened_evidence':out})
   self.assertEqual(expand_source_headers(packed)['opened_evidence'],out)
 def test_no_inference_from_s2(self):
  docs=[dict(source_id='S2:a#c0',text='Abstract')]
  self.assertEqual(pdf_scenario_evidence(docs,{}),docs)
 def test_no_fake_success_from_abstract(self):
  with self.assertRaises(ValueError):
   pdf_scenario_evidence([dict(source_id='S2:a#c0',text='Abstract',content_level='pubmed_abstract')],dict(kind='dual_pdf_s2',synthetic=True,real_execution=False,id_mapping={'PMID:1':'S2:a'}))

if __name__=='__main__':unittest.main()
