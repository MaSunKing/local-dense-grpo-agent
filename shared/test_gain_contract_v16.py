import copy
import unittest

from gain_contract import canonical_action, digest, validate_tool_execution


class ActionParserTests(unittest.TestCase):
    def test_runtime_xml_search_and_browse_are_canonically_projected(self):
        self.assertEqual(canonical_action(
            '<think>private reasoning</think><call_tool name="pubmed_search">heart failure</call_tool>',
            'xml_call_tool_v1'),
            {'tool':'search','arguments':{'query':'heart failure'}})
        self.assertEqual(canonical_action(
            '<call_tool name="browse_document" query="renal outcomes">PMID:1</call_tool>',
            'xml_call_tool_v1'),
            {'tool':'browse','arguments':{'candidate_id':'PMID:1','focus':'renal outcomes'}})
        self.assertEqual(canonical_action(
            '<call_tool name="browse_webpage">WEB:1</call_tool>',
            'xml_call_tool_v1'),
            {'tool':'browse','arguments':{'candidate_id':'WEB:1','focus':''}})

    def test_xml_unknown_attributes_fail_closed(self):
        with self.assertRaises(ValueError):
            canonical_action(
                '<call_tool name="browse_document" query="renal" future_final="secret">PMID:1</call_tool>',
                'xml_call_tool_v1')

    def test_tool_execution_binds_original_xml_without_rewriting_it(self):
        raw='<call_tool name="browse_webpage" query="outcome">WEB:1</call_tool>'
        action=canonical_action(raw,'xml_call_tool_v1')
        response={'tool':'browse','evidence':[{'source_id':'WEB:1','text':'Observed evidence.'}]}
        core={'version':'trusted_tool_execution_v1','question_id':'q','rollout_id':'r',
              'record_id':'d','policy':'p','token_digest':'t','task_scope_digest':'s',
              'capture':{'capture_id':'c','capture_sha256':'a'*64,
                         'parser_version':'xml_call_tool_v1','raw_completion':raw,
                         'action':action,'action_digest':digest(action)},
              'response':{'payload':response,'payload_digest':digest(response)},
              'status':'succeeded'}
        execution=copy.deepcopy(core);execution['tool_execution_id']=digest(core)
        parsed,_=validate_tool_execution(execution)
        self.assertEqual(parsed,action)
        self.assertEqual(execution['capture']['raw_completion'],raw)


if __name__ == '__main__':
    unittest.main()
