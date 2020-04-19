import unittest
from data_loader import DataLoader
from util import load_documents, index_document_entities, output_pred_dist
from util import use_cuda, save_model, load_model, get_config, load_dict, cal_accuracy
import json

class DataTest(unittest.TestCase):
    # def init(self):
    #     cfg = get_config("config/webqsp.yml")

    def setUp(self):
        self.cfg = get_config("config/webqsp.yml")
        # data_folder: 'datasets/webqsp/full/'
        # entity2id: 'entities.txt'
        # entity_example:
    def sliceDict(self,example):
        return {k:example[k] for k in list(example.keys())[:5]}

    # < fb: m.01t2w8 >
    # < fb: m.010fm02b >
    # < fb: m.0c94x >

    def testEntity(self):
        cfg = self.cfg
        entity2id = load_dict(cfg['data_folder'] + cfg['entity2id'])

    # {'<fb:aviation.aircraft_manufacturer.aircraft_models_made>': 0,
    #  '<fb:medicine.notable_person_with_medical_condition.condition>': 1,
    #  '<fb:music.group_membership.group>': 2}
    def testRelation(self):
        cfg = self.cfg
        relation2id = load_dict(cfg['data_folder'] + cfg['relation2id'])
        print(self.sliceDict(relation2id))

    def testLoad(self,cfg):
        cfg = self.cfg
        entity2id = load_dict(cfg['data_folder'] + cfg['entity2id'])
        word2id = load_dict(cfg['data_folder'] + cfg['word2id'])
        relation2id = load_dict(cfg['data_folder'] + cfg['relation2id'])

        train_documents = load_documents(cfg['data_folder'] + cfg['train_documents'])
        train_document_entity_indices, train_document_texts = index_document_entities(train_documents, word2id,
                                                                                      entity2id,
                                                                                      cfg['max_document_word'])
        train_data = DataLoader(cfg['data_folder'] + cfg['train_data'], train_documents, train_document_entity_indices,
                                train_document_texts, word2id, relation2id, entity2id, cfg['max_query_word'],
                                cfg['max_document_word'], cfg['use_kb'], cfg['use_doc'], cfg['use_inverse_relation'])

    def testTrainData(self):
        lines = open(self.cfg["data_folder"] + self.cfg['train_data'], encoding='utf-8').readlines()[0]
        print(json.loads(lines))


    def testLoadEntity(self):
        entity2id = load_dict(self.cfg['data_folder'] + self.cfg['entity2id'])
        example = {k: entity2id[k] for k in list(entity2id.keys())[:5]}
        print(example)

    def testLoadDoc(self):
        print(load_documents(self.cfg["data_folder"] + self.cfg['train_documents'])[:3])

    # def tearDown(self):
