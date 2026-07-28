import re

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, BRICS, Draw, rdMolDescriptors
from rdkit.Chem import rdFMCS
import networkx as nx
from torch_geometric.data import Data, Batch, HeteroData
from torch_geometric.nn import global_mean_pool, SAGPooling
from torch_geometric.utils import to_networkx
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import math
import random
from collections import defaultdict, deque
from torch.utils.data import DataLoader, Dataset
import warnings

warnings.filterwarnings('ignore')

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


class ElementKnowledgeGraph:
    """元素知识图谱 """

    def __init__(self):
        # 三层结构：类级别 -> 元素/官能团级别 -> 属性级别
        # 官能团SMARTS模式
        self.functional_group_smarts = {
            "Alkyl": "[CX4]",
            "Alkenyl": "[$([CX3]=[CX3])]",
            "Alkynyl": "[$([CX2]#C)]",
            "Phenyl": "c",
            "Bromoalkane": "[Br]",
            "Chloro": "[Cl]",
            "Fluoro": "[F]",
            "Halo": "[#6][F,Cl,Br,I]",
            "Iodo": "[I]",
            "Acetal": "O[CH1][OX2H0]",
            "Haloformyl": "[CX3](=[OX1])[F,Cl,Br,I]",
            "Hydroxyl": "[#6][OX2H]",
            "Aldehyde": "[CX3H1](=O)[#6]",
            "CarbonateEster": "[CX3](=[OX1])(O)O",
            "Carboxylate": "[CX3](=O)[O-]",
            "Carboxyl": "[CX3](=O)[OX2H1]",
            "Carboalkoxy": "[CX3](=O)[OX2H0]",
            "Ether": "[OD2]",
            "Hemiacetal": "O[CH1][OX2H1]",
            "Hemiketal": "OC[OX2H1]",
            "Methylenedioxy": "C([OX2])([OX2])",
            "Hydroperoxy": "O[OX2H]",
            "Ketal": "OC[OX2H0]",
            "Carbonyl": "[CX3]=[OX1]",
            "CarboxylicAnhydride": "[CX3](=O)[OX2H0][CX3](=O)",
            "OrthocarbonateEster": "C([OX2])([OX2])([OX2])([OX2])",
            "Orthoester": "C([OX2])([OX2])([OX2])",
            "Peroxy": "O[OX2H0]",
            "Carboxamide": "[NX3][CX3](=[OX1])[#6]",
            "Amidine": "[NX3][CX3]=[NX2]",
            "AmmoniumIon": "[NX4+]",
            "PrimaryAmine": "[NX3;H2,H1;!$(NC=O)]",
            "SecondaryAmine": "[NX3;H2;!$(NC=O)]",
            "TertiaryAmine": "[NX3;!$(NC=O)]",
            "Azide": "[$(*-[NX2-]-[NX2+]#[NX1]),$(*-[NX2]=[NX2+]=[NX1-])]",
            "Azo": "[NX2]=N",
            "Carbamate": "[NX3,NX4+][CX3](=[OX1])[OX2,OX1-]",
            "Cyanate": "OC#N",
            "Isocyanate": "[O]=[CX2]=[NX2]",
            "Imide": "[CX3](=[OX1])[NX3H][CX3](=[OX1])",
            "PrimaryAldimine": "[CX3H1]=[NX2H1]",
            "PrimaryKetimine": "[CX3]=[NX2H1]",
            "SecondaryAldimine": "[CX3H1]=[NX2H0]",
            "SecondaryKetimine": "[CX3]=[NX2H0]",
            "Nitrate": "[$([NX3](=[OX1])(=[OX1])O),$([NX3+]([OX1-])(=[OX1])O)]",
            "Isonitrile": "[CX1-]#[NX2+]",
            "Nitrile": "[NX1]#[CX2]",
            "Nitrosooxy": "O[NX2]=[OX1]",
            "Nitro": "[$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8]",
            "Nitroso": "[NX2]=[OX1]",
            "Oxime": "C=N[OX2H1]",
            "Pyridyl": "ccccnc",
            "Disulfide": "[#16X2H0]S",
            "CarbodithioicAcid": "[#16X2H1]C=[#16]",
            "Carbodithio": "[#16X2H0]C=[#16]",
            "Sulfide": "[#16X2H0]",
            "Sulfino": "[$([#16X3](=[OX1])[OX2H,OX1H0-]),$([#16X3+]([OX1-])[OX2H,OX1H0-])]",
            "Sulfoate": "[$([#16X4](=[OX1])(=[OX1])([#6])[OX2H0]),$([#16X4+2]([OX1-])([OX1-])([#6])[OX2H0])]",
            "Sulfonyl": "[$([#16X4](=[OX1])(=[OX1])([#6])[#6]),$([#16X4+2]([OX1-])([OX1-])([#6])[#6])]",
            "Sulfo": "[$([#16X4](=[OX1])(=[OX1])([#6])[OX2H,OX1H0-]),$([#16X4+2]([OX1-])([OX1-])([#6])[OX2H,OX1H0-])]",
            "Sulfinyl": "[$([#16X3]=[OX1]),$([#16X3+][OX1-])]",
            "Thial": "[#16]=[CX3H1]",
            "CarbothioicOAcid": "[OX2H1]C=[#16]",
            "CarbothioicSAcid": "[#16X2H1]C=O",
            "Isothiocyanate": "[#16]=[CX2]=[NX2]",
            "Thiocyanate": "[#16]C#N",
            "Thiolester": "[#16X2H0]C=O",
            "Thionoester": "[OX2H0]C=[#16]",
            "Thioketone": "[#16]=[CX3H0]",
            "Sulfhydryl": "[#16X2H]",
            "Phosphate": "[OX2H0][PX4](=[OX1])([OX2H1])([OX2H1])",
            "Phosphino": "[PX3]",
            "Phosphodiester": "[OX2H1][PX4](=[OX1])([OX2H0])([OX2H0])",
            "Phosphono": "[PX4](=[OX1])([OX2H1])([OX2H1])",
            "Borino": "[BX3]([OX2H1])",
            "Borinate": "[BX3]([OX2H0])",
            "Borono": "[BX3]([OX2H1])([OX2H1])",
            "Boronate": "[BX3]([OX2H0])([OX2H0])",
            "Alkylaluminium": "[#13].[#13]",
            "Alkyllithium": "[#3]",
            "AlkylmagnesiumHalide": "[#12X2][F,Cl,Br,I]",
            "SilylEther": "[#14X4][OX2]"
        }

        self.FUNCTIONAL_GROUP_ELEMENTS, self.element_in_group = self.smarts_to_elements()

        self.entities = {}
        self.relations = []
        self.hierarchy = {
            'class': ['Nonmetals', 'Metals', 'Halogens', 'NobleGases'],
            'element': ['C', 'N', 'O', 'H', 'S', 'P', 'F', 'Cl', 'Br', 'I'],
            'functional_group': list(self.functional_group_smarts.keys()),
            'property': ['Weight1', 'Weight2', 'Weight3', 'Weight4', 'Weight5',
                         'Period1', 'Period2', 'Period3', 'Period4', 'Period5']
        }

        # 元素属性映射
        self.element_properties = {
            'C': {'class': 'Nonmetals', 'weight': 'Weight2', 'period': 'Period2'},
            'N': {'class': 'Nonmetals', 'weight': 'Weight2', 'period': 'Period2'},
            'O': {'class': 'Nonmetals', 'weight': 'Weight2', 'period': 'Period2'},
            'H': {'class': 'Nonmetals', 'weight': 'Weight1', 'period': 'Period1'},
            'S': {'class': 'Nonmetals', 'weight': 'Weight3', 'period': 'Period3'},
            'P': {'class': 'Nonmetals', 'weight': 'Weight3', 'period': 'Period3'},
            'F': {'class': 'Halogens', 'weight': 'Weight2', 'period': 'Period2'},
            'Cl': {'class': 'Halogens', 'weight': 'Weight3', 'period': 'Period3'},
            'Br': {'class': 'Halogens', 'weight': 'Weight4', 'period': 'Period4'},
            'I': {'class': 'Halogens', 'weight': 'Weight5', 'period': 'Period5'}
        }

        # 构建知识图谱的三元组
        self.triples = []

        # 类级别关系
        self.triples.extend([
            ('ReactiveNonmetal', 'isSubClassOf', 'Nonmetals'),
            ('AlkaliMetal', 'isSubClassOf', 'Metals'),
            ('TransitionMetal', 'isSubClassOf', 'Metals')
        ])

        # 元素和官能团关系
        for fg, elements in self.FUNCTIONAL_GROUP_ELEMENTS.items():

            for ele in elements:
                self.triples.append(
                    (ele, "isPartOf", fg)
                )

        # 属性关系
        self.triples.extend([
            ('C', 'hasWeight', 'Weight2'),
            ('N', 'hasWeight', 'Weight2'),
            ('O', 'hasWeight', 'Weight2'),
            ('H', 'hasWeight', 'Weight1'),
            ('S', 'hasWeight', 'Weight3'),
            ('P', 'hasWeight', 'Weight3'),
            ('F', 'hasWeight', 'Weight2'),
            ('Cl', 'hasWeight', 'Weight3'),
            ('Br', 'hasWeight', 'Weight4'),
            ('I', 'hasWeight', 'Weight5'),
            ('H', 'isinPeriod', 'Period1'),
            ('C', 'isinPeriod', 'Period2'),
            ('N', 'isinPeriod', 'Period2'),
            ('O', 'isinPeriod', 'Period2'),
            ('F', 'isinPeriod', 'Period2'),
            ('P', 'isinPeriod', 'Period3'),
            ('S', 'isinPeriod', 'Period3'),
            ('Cl', 'isinPeriod', 'Period3'),
            ('Br', 'isinPeriod', 'Period4'),
            ('I', 'isinPeriod', 'Period5')
        ])



    def get_element_features(self, symbol):
        """获取元素特征向量"""
        if symbol in self.element_properties:
            elem = self.element_properties[symbol]
            # 创建特征向量 [class_idx, weight_idx, period_idx]
            class_mapping = {'Nonmetals': 0, 'Halogens': 1, 'Metals': 2}
            weight_mapping = {'Weight1': 0, 'Weight2': 1, 'Weight3': 2, 'Weight4': 3, 'Weight5': 4}
            period_mapping = {'Period1': 0, 'Period2': 1, 'Period3': 2, 'Period4': 3, 'Period5': 4}

            return torch.tensor([
                class_mapping.get(elem['class'], 0),
                weight_mapping.get(elem['weight'], 0),
                period_mapping.get(elem['period'], 0)
            ], dtype=torch.float)
        else:
            return torch.zeros(3, dtype=torch.float)

    def get_functional_groups(self, mol):
        """检测分子中的官能团"""
        functional_groups = []
        for name, smarts in self.functional_group_smarts.items():
            pattern = Chem.MolFromSmarts(smarts)
            if pattern is not None:
                matches = mol.GetSubstructMatches(pattern)
                if matches:
                    functional_groups.append((name, matches))
        return functional_groups

    def get_2hop_connections(self, entity):
        """获取2跳连接 """
        connections = []
        visited = set()
        queue = deque([(entity, 0, [])])

        while queue:
            current, hops, path = queue.popleft()
            if hops > 2:
                continue

            if hops == 2 and current != entity:
                connections.append((entity, current, path))
                continue

            for triple in self.triples:
                if triple[0] == current and triple[1] not in visited:
                    visited.add(triple[1])
                    queue.append((triple[2], hops + 1, path + [triple[1]]))
                elif triple[2] == current and triple[1] not in visited:
                    visited.add(triple[1])
                    queue.append((triple[0], hops + 1, path + [triple[1]]))

        return connections

    def smarts_to_elements(self):
        s = {}
        s1 = {}
        for name, smart in self.functional_group_smarts.items():
            atoms = self.get_elements_from_smarts(smart)
            s[name] = atoms
            s1[name] = list(atoms)
        return s, s1

    def get_elements_from_smarts(self, smarts):
        """
        从SMARTS模板中自动提取涉及的元素
        支持：
            C N O P S H
            Cl Br F I
            Si B Li Mg Al ...
            #6 #7 #8 #16 ...
        """
        ptable = Chem.GetPeriodicTable()
        elements = set()

        nums = re.findall(r'#(\d+)', smarts)

        for num in nums:
            try:
                elements.add(
                    ptable.GetElementSymbol(int(num))
                )
            except:
                pass
        atoms = re.findall(
            r'Cl|Br|Si|Mg|Li|Al|Na|Ca|Fe|Zn|Cu|Ag|Hg|Pb|Sn|B|C|N|O|P|S|F|I|H',
            smarts
        )

        elements.update(atoms)

        return elements


# 其余的类实现保持不变...
class MolecularGraphBuilder:
    """分子图构建器 """

    def __init__(self):
        self.element_kg = ElementKnowledgeGraph()

    def smiles_to_molecule_graph(self, smiles):
        """构建分子视图 - 包含原子、键、片段"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # 添加氢原子
        mol = Chem.AddHs(mol)

        # 获取原子特征
        atom_features = []
        atom_symbols = []
        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            atom_symbols.append(symbol)
            features = [
                atom.GetAtomicNum(),
                atom.GetDegree(),
                atom.GetFormalCharge(),
                int(atom.GetChiralTag()),
                int(atom.GetHybridization()),
                int(atom.GetIsAromatic()),
                atom.GetTotalNumHs(),
                atom.GetMass()
            ]
            atom_features.append(features)

        atom_features = torch.tensor(atom_features, dtype=torch.float)

        # 获取键信息
        edge_index = []
        edge_attr = []
        bond_types = []

        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            # 双向边
            edge_index.append([i, j])
            edge_index.append([j, i])

            bond_type = int(bond.GetBondType())
            bond_features = [
                bond_type,
                int(bond.GetIsConjugated()),
                int(bond.IsInRing()),
                int(bond.GetStereo())
            ]
            edge_attr.append(bond_features)
            edge_attr.append(bond_features)
            bond_types.extend([bond_type, bond_type])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float) if edge_attr else torch.empty((0, 4), dtype=torch.float)

        # 使用BRICS进行分子碎片化
        fragments = self._brics_fragmentation(mol)

        # 构建异构数据
        data = HeteroData()

        # 添加原子节点
        data['atom'].x = atom_features
        data['atom'].symbols = atom_symbols

        # 添加片段节点
        fragment_features = []
        for frag in fragments:
            # 片段特征：原子数、重原子数、分子量等
            frag_mol = Chem.MolFromSmiles(Chem.MolToSmiles(frag))
            features = [
                frag_mol.GetNumAtoms(),
                frag_mol.GetNumHeavyAtoms(),
                rdMolDescriptors.CalcExactMolWt(frag_mol),
                len(frag_mol.GetRingInfo().AtomRings())
            ]
            fragment_features.append(features)

        if fragment_features:
            data['fragment'].x = torch.tensor(fragment_features, dtype=torch.float)
        else:
            data['fragment'].x = torch.empty((0, 4), dtype=torch.float)

        # 添加原子-原子边（化学键）
        data['atom', 'bond', 'atom'].edge_index = edge_index
        data['atom', 'bond', 'atom'].edge_attr = edge_attr

        # 添加片段-反应-片段边
        fragment_edges = self._get_fragment_reaction_edges(fragments)
        if fragment_edges:
            fragment_edge_index = torch.tensor(fragment_edges, dtype=torch.long).t().contiguous()
            fragment_edge_attr = torch.ones(fragment_edge_index.size(1), 1)  # 反应类型特征
            data['fragment', 'reaction', 'fragment'].edge_index = fragment_edge_index
            data['fragment', 'reaction', 'fragment'].edge_attr = fragment_edge_attr

        # 添加原子-连接-片段边
        atom_fragment_edges = self._get_atom_fragment_edges(mol, fragments)
        if atom_fragment_edges:
            atom_frag_edge_index = torch.tensor(atom_fragment_edges, dtype=torch.long).t().contiguous()
            atom_frag_edge_attr = torch.ones(atom_frag_edge_index.size(1), 1)  # 连接类型特征
            data['atom', 'join', 'fragment'].edge_index = atom_frag_edge_index
            data['atom', 'join', 'fragment'].edge_attr = atom_frag_edge_attr

        data.smiles = smiles
        data.mol = mol
        data.fragments = fragments

        return data

    def _brics_fragmentation(self, mol):
        """使用BRICS算法进行分子碎片化"""
        try:
            # 使用BRICS分解分子
            brics_bonds = list(BRICS.FindBRICSBonds(mol))
            if not brics_bonds:
                return [mol]

            # 获取断键索引
            break_bonds = [bond[0] for bond in brics_bonds]

            # 断键并获取片段
            fragmented_mol = BRICS.BreakBRICSBonds(mol, break_bonds)
            fragments = list(fragmented_mol)

            return fragments
        except:
            # 如果BRICS失败，返回整个分子作为一个片段
            return [mol]

    def _get_fragment_reaction_edges(self, fragments):
        """获取片段-反应-片段边"""
        edges = []
        # 如果两个片段共享原子模式，则认为它们有反应关系
        for i in range(len(fragments)):
            for j in range(i + 1, len(fragments)):
                frag1_smiles = Chem.MolToSmiles(fragments[i])
                frag2_smiles = Chem.MolToSmiles(fragments[j])

                # 检查是否有共同的反应位点
                if self._has_common_reaction_site(fragments[i], fragments[j]):
                    edges.append([i, j])
                    edges.append([j, i])

        return edges

    def _has_common_reaction_site(self, frag1, frag2):
        """检查两个片段是否有共同的反应位点"""
        try:
            # 使用最大公共子结构作为简化判断
            mcs = rdFMCS.FindMCS([frag1, frag2], timeout=1)
            if mcs.numAtoms > 0:
                return True
        except:
            pass
        return False

    def _get_atom_fragment_edges(self, mol, fragments):
        """获取原子-片段连接边"""
        edges = []
        atom_fragment_map = {}

        # 构建原子到片段的映射
        for frag_idx, fragment in enumerate(fragments):
            frag_mol = Chem.MolFromSmiles(Chem.MolToSmiles(fragment))
            if frag_mol is None:
                continue

            # 在原始分子中查找片段匹配
            matches = mol.GetSubstructMatches(frag_mol)
            if matches:
                for match in matches:
                    for atom_idx in match:
                        if atom_idx not in atom_fragment_map:
                            atom_fragment_map[atom_idx] = []
                        atom_fragment_map[atom_idx].append(frag_idx)

        # 构建边
        for atom_idx, frag_indices in atom_fragment_map.items():
            for frag_idx in frag_indices:
                edges.append([atom_idx, frag_idx])

        return edges


class MultiViewGraphBuilder:
    """多视图图构建器 - 严格按照论文实现三个视图"""

    def __init__(self):
        self.mol_builder = MolecularGraphBuilder()

    def build_molecule_view(self, smiles):
        """构建分子视图 - 包含两个节点类型和三个边类型"""
        return self.mol_builder.smiles_to_molecule_graph(smiles)

    def build_element_view(self, smiles):
        """构建元素视图 - 在分子视图基础上添加两个节点类型和五个边类型"""
        mol_data = self.build_molecule_view(smiles)
        if mol_data is None:
            return None

        mol = mol_data.mol

        # 创建元素视图的异构数据
        element_data = HeteroData()

        # 复制分子视图的所有内容
        for node_type in mol_data.node_types:
            element_data[node_type].x = mol_data[node_type].x.clone()
            if hasattr(mol_data[node_type], 'symbols'):
                element_data[node_type].symbols = mol_data[node_type].symbols.copy()

        for edge_type in mol_data.edge_types:
            element_data[edge_type].edge_index = mol_data[edge_type].edge_index.clone()
            if hasattr(mol_data[edge_type], 'edge_attr'):
                element_data[edge_type].edge_attr = mol_data[edge_type].edge_attr.clone()

        element_data.smiles = mol_data.smiles
        element_data.mol = mol_data.mol
        element_data.fragments = mol_data.fragments

        # 添加元素知识图谱节点
        element_nodes = set()
        functional_group_nodes = set()

        # 1. 原子-元素边 (Atom-AE-Element)
        atom_element_edges = []
        for i, symbol in enumerate(mol_data['atom'].symbols):
            element_nodes.add(symbol)
            atom_element_edges.append([i, list(element_nodes).index(symbol)])

        # 添加元素节点特征
        if element_nodes:
            element_features = []
            for elem in element_nodes:
                elem_feat = self.mol_builder.element_kg.get_element_features(elem)
                element_features.append(elem_feat)
            element_data['element'].x = torch.stack(element_features)

        # 添加原子-元素边
        if atom_element_edges:
            atom_element_edge_index = torch.tensor(atom_element_edges, dtype=torch.long).t().contiguous()
            atom_element_edge_attr = torch.ones(atom_element_edge_index.size(1), 1)
            element_data['atom', 'AE', 'element'].edge_index = atom_element_edge_index
            element_data['atom', 'AE', 'element'].edge_attr = atom_element_edge_attr

        # 2. 片段-官能团边 (Fragment-FrFu-Functional Group)
        fragment_functional_edges = []
        functional_groups = self.mol_builder.element_kg.get_functional_groups(mol)

        for func_name, matches in functional_groups:
            functional_group_nodes.add(func_name)
            func_idx = list(functional_group_nodes).index(func_name)

            # 找到包含该官能团的片段
            for frag_idx, fragment in enumerate(mol_data.fragments):
                frag_smiles = Chem.MolToSmiles(fragment)
                frag_mol = Chem.MolFromSmiles(frag_smiles)
                if frag_mol and self._contains_functional_group(frag_mol, func_name):
                    fragment_functional_edges.append([frag_idx, func_idx])

        # 添加官能团节点特征
        if functional_group_nodes:
            func_features = []
            for func in functional_group_nodes:
                # 官能团特征
                features = [len(func), hash(func) % 100]
                func_features.append(features)
            element_data['functional_group'].x = torch.tensor(func_features, dtype=torch.float)

        # 添加片段-官能团边
        if fragment_functional_edges:
            frag_func_edge_index = torch.tensor(fragment_functional_edges, dtype=torch.long).t().contiguous()
            frag_func_edge_attr = torch.ones(frag_func_edge_index.size(1), 1)
            element_data['fragment', 'FrFu', 'functional_group'].edge_index = frag_func_edge_index
            element_data['fragment', 'FrFu', 'functional_group'].edge_attr = frag_func_edge_attr

        # 3. 元素-元素边 (Element-EE-Element) - 2跳连接
        element_element_edges = set()
        for elem in element_nodes:
            connections = self.mol_builder.element_kg.get_2hop_connections(elem)
            for _, target, _ in connections:
                if target in element_nodes:
                    src_idx = list(element_nodes).index(elem)
                    tgt_idx = list(element_nodes).index(target)
                    element_element_edges.add((src_idx, tgt_idx))
                    element_element_edges.add((tgt_idx, src_idx))

        if element_element_edges:
            ee_edge_index = torch.tensor(list(element_element_edges), dtype=torch.long).t().contiguous()
            ee_edge_attr = torch.ones(ee_edge_index.size(1), 1)
            element_data['element', 'EE', 'element'].edge_index = ee_edge_index
            element_data['element', 'EE', 'element'].edge_attr = ee_edge_attr

        # 4. 官能团-官能团边 (Functional Group-FuFu-Functional Group) - 2跳连接
        func_func_edges = set()
        for func in functional_group_nodes:
            # 简化实现：相似官能团之间建立连接
            for other_func in functional_group_nodes:
                if func != other_func and self._are_functional_groups_similar(func, other_func):
                    src_idx = list(functional_group_nodes).index(func)
                    tgt_idx = list(functional_group_nodes).index(other_func)
                    func_func_edges.add((src_idx, tgt_idx))

        if func_func_edges:
            ff_edge_index = torch.tensor(list(func_func_edges), dtype=torch.long).t().contiguous()
            ff_edge_attr = torch.ones(ff_edge_index.size(1), 1)
            element_data['functional_group', 'FuFu', 'functional_group'].edge_index = ff_edge_index
            element_data['functional_group', 'FuFu', 'functional_group'].edge_attr = ff_edge_attr

        # 5. 元素-官能团边 (Element-EFu-Functional Group)
        element_func_edges = []
        for elem in element_nodes:
            for func in functional_group_nodes:
                # 检查元素是否是该官能团的组成部分
                if self._is_element_in_functional_group(elem, func):
                    elem_idx = list(element_nodes).index(elem)
                    func_idx = list(functional_group_nodes).index(func)
                    element_func_edges.append([elem_idx, func_idx])

        if element_func_edges:
            ef_edge_index = torch.tensor(element_func_edges, dtype=torch.long).t().contiguous()
            ef_edge_attr = torch.ones(ef_edge_index.size(1), 1)
            element_data['element', 'EFu', 'functional_group'].edge_index = ef_edge_index
            element_data['element', 'EFu', 'functional_group'].edge_attr = ef_edge_attr

        return element_data

    def build_drug_view(self, smiles, drug_features=None):
        """构建药物视图 - 在分子视图基础上添加一个节点类型和两个边类型"""
        mol_data = self.build_molecule_view(smiles)
        if mol_data is None:
            return None

        # 创建药物视图的异构数据
        drug_data = HeteroData()

        # 复制分子视图的所有内容
        for node_type in mol_data.node_types:
            drug_data[node_type].x = mol_data[node_type].x.clone()
            if hasattr(mol_data[node_type], 'symbols'):
                drug_data[node_type].symbols = mol_data[node_type].symbols.copy()

        for edge_type in mol_data.edge_types:
            drug_data[edge_type].edge_index = mol_data[edge_type].edge_index.clone()
            if hasattr(mol_data[edge_type], 'edge_attr'):
                drug_data[edge_type].edge_attr = mol_data[edge_type].edge_attr.clone()

        drug_data.smiles = mol_data.smiles
        drug_data.mol = mol_data.mol
        drug_data.fragments = mol_data.fragments

        # 添加DNode节点
        if drug_features is not None:
            drug_data['dnode'].x = drug_features.unsqueeze(0)  # [1, feature_dim]
        else:
            # 如果没有提供药物特征，使用零向量
            drug_data['dnode'].x = torch.rand(1, 128, dtype=torch.float)

        # 1. 原子-DNode边 (Atom-AD-DNode)
        num_atoms = mol_data['atom'].x.size(0)
        atom_dnode_edges = [[i, 0] for i in range(num_atoms)]  # 所有原子连接到DNode

        atom_dnode_edge_index = torch.tensor(atom_dnode_edges, dtype=torch.long).t().contiguous()
        atom_dnode_edge_attr = torch.ones(atom_dnode_edge_index.size(1), 1)
        drug_data['atom', 'AD', 'dnode'].edge_index = atom_dnode_edge_index
        drug_data['atom', 'AD', 'dnode'].edge_attr = atom_dnode_edge_attr

        # 2. 片段-DNode边 (Fragment-FrD-DNode)
        num_fragments = mol_data['fragment'].x.size(0) if hasattr(mol_data['fragment'], 'x') else 0
        fragment_dnode_edges = [[i, 0] for i in range(num_fragments)]  # 所有片段连接到DNode

        if fragment_dnode_edges:
            frag_dnode_edge_index = torch.tensor(fragment_dnode_edges, dtype=torch.long).t().contiguous()
            frag_dnode_edge_attr = torch.ones(frag_dnode_edge_index.size(1), 1)
            drug_data['fragment', 'FrD', 'dnode'].edge_index = frag_dnode_edge_index
            drug_data['fragment', 'FrD', 'dnode'].edge_attr = frag_dnode_edge_attr

        return drug_data

    def _contains_functional_group(self, mol, functional_group_name):
        """检查分子是否包含指定的官能团"""
        smarts = self.mol_builder.element_kg.functional_group_smarts.get(functional_group_name)
        if smarts:
            pattern = Chem.MolFromSmarts(smarts)
            return mol.HasSubstructMatch(pattern) if pattern else False
        return False

    def _are_functional_groups_similar(self, func1, func2):
        """检查两个官能团是否相似"""

        """
            判断两个官能团是否相似
            基于组成元素的Jaccard Similarity
            """
        # 官能团组成元素
        FUNCTIONAL_GROUP_ELEMENTS = {
            'Alcohol': {'O', 'C', 'H'},
            'Carbonyl': {'C', 'O'},
            'Carboxyl': {'C', 'O', 'H'},
            'Amino': {'N', 'H'},
            'Amide': {'C', 'O', 'N'},
            'Ester': {'C', 'O'},
            'Ether': {'O', 'C'},
            'Nitrile': {'C', 'N'},
            'Nitro': {'N', 'O'},
            'Sulfoxide': {'S', 'O'},
            'Sulfone': {'S', 'O'}
        }

        elems1 = self.mol_builder.element_kg.FUNCTIONAL_GROUP_ELEMENTS.get(func1, set())
        elems2 = self.mol_builder.element_kg.FUNCTIONAL_GROUP_ELEMENTS.get(func2, set())

        if len(elems1) == 0 or len(elems2) == 0:
            return False

        intersection = len(elems1 & elems2)
        union = len(elems1 | elems2)

        similarity = intersection / union

        return similarity >= 0.8


    def _is_element_in_functional_group(self, element, functional_group):
        """检查元素是否是官能团的组成部分"""
        # 基于预定义的知识
        element_in_group = {
            'Alcohol': ['O', 'C', 'H'],
            'Carbonyl': ['C', 'O'],
            'Carboxyl': ['C', 'O', 'H'],
            'Amino': ['N', 'H'],
            'Amide': ['C', 'O', 'N'],
            'Ester': ['C', 'O'],
            'Ether': ['O', 'C'],
            'Nitrile': ['C', 'N'],
            'Nitro': ['N', 'O'],
            'Sulfoxide': ['S', 'O'],
            'Sulfone': ['S', 'O']
        }

        return element in element_in_group.get(functional_group, [])


class HeteroGNNEncoder(nn.Module):
    """异构图表编码器 - 处理多种节点类型和边类型"""

    def __init__(self, node_dims, edge_dims, hidden_dim=256, n_layers=3, n_heads=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # 节点类型特定的投影层
        self.node_projs = nn.ModuleDict()
        self.node_dims = node_dims

        # 边类型特定的投影层
        self.edge_projs = nn.ModuleDict()
        self.edge_dims = edge_dims

        # 异构消息传递层
        self.layers = nn.ModuleList([
            HeteroMessagePassingLayer(hidden_dim, n_heads)
            for _ in range(n_layers)
        ])

        # 图池化
        self.pool = SAGPooling(hidden_dim, ratio=0.5)

        # 存储设备信息
        self._device = None

    @staticmethod
    def edge_type_to_str(edge_type):
        if isinstance(edge_type, tuple):
            return '__'.join(edge_type)
        else:
            return str(edge_type)

    def to(self, device):
        """重写to方法，确保所有组件都移动到设备"""
        super().to(device)
        self._device = device

        # 确保所有现有的投影层也在设备上
        for node_type in list(self.node_projs.keys()):
            self.node_projs[node_type] = self.node_projs[node_type].to(device)

        for edge_type_str in list(self.edge_projs.keys()):
            self.edge_projs[edge_type_str] = self.edge_projs[edge_type_str].to(device)

        return self

    def _ensure_projection_layers(self, hetero_data):
        """确保所有节点和边类型都有对应的投影层"""
        # 获取设备
        if self._device is None:
            self._device = next(self.parameters()).device

        # 处理 SimpleHeteroWrapper 的情况
        if hasattr(hetero_data, 'hetero_data_list') and hetero_data.hetero_data_list:
            hetero_data = hetero_data.hetero_data_list[0]

        # 确保节点投影层
        for node_type in hetero_data.node_types:
            if hasattr(hetero_data[node_type], 'x') and hetero_data[node_type].x is not None:
                actual_dim = hetero_data[node_type].x.size(1)

                if node_type not in self.node_projs:
                    self.node_projs[node_type] = nn.Linear(actual_dim, self.hidden_dim).to(self._device)

        # 确保边投影层
        for edge_type in hetero_data.edge_types:
            edge_store = hetero_data[edge_type]
            if hasattr(edge_store, 'edge_attr') and edge_store.edge_attr is not None:
                edge_type_str = self.edge_type_to_str(edge_type)
                actual_dim = edge_store.edge_attr.size(1)

                if edge_type_str not in self.edge_projs:
                    self.edge_projs[edge_type_str] = nn.Linear(actual_dim, self.hidden_dim).to(self._device)

    def forward(self, hetero_data, target_node_type='atom'):
        # 确保所有投影层都存在且维度正确
        self._ensure_projection_layers(hetero_data)

        # 处理 SimpleHeteroWrapper 的情况
        if hasattr(hetero_data, 'hetero_data_list') and hetero_data.hetero_data_list:
            hetero_data = hetero_data.hetero_data_list[0]

        # 投影所有节点特征到统一空间
        x_dict = {}
        for node_type in hetero_data.node_types:
            if hasattr(hetero_data[node_type], 'x') and hetero_data[node_type].x is not None:
                # 确保节点特征在正确的设备上
                if hetero_data[node_type].x.device != self._device:
                    hetero_data[node_type].x = hetero_data[node_type].x.to(self._device)

                if node_type in self.node_projs:
                    x_dict[node_type] = self.node_projs[node_type](hetero_data[node_type].x)
                else:
                    # 使用默认投影
                    actual_dim = hetero_data[node_type].x.size(1)
                    self.node_projs[node_type] = nn.Linear(actual_dim, self.hidden_dim).to(self._device)
                    x_dict[node_type] = self.node_projs[node_type](hetero_data[node_type].x)

        # 投影所有边特征到统一空间
        edge_attr_dict = {}
        for edge_type in hetero_data.edge_types:
            edge_store = hetero_data[edge_type]
            if hasattr(edge_store, 'edge_attr') and edge_store.edge_attr is not None:
                # 确保边特征在正确的设备上
                if edge_store.edge_attr.device != self._device:
                    edge_store.edge_attr = edge_store.edge_attr.to(self._device)

                edge_type_str = self.edge_type_to_str(edge_type)
                if edge_type_str in self.edge_projs:
                    edge_attr_dict[edge_type] = self.edge_projs[edge_type_str](edge_store.edge_attr)
                else:
                    # 使用默认投影
                    actual_dim = edge_store.edge_attr.size(1)
                    self.edge_projs[edge_type_str] = nn.Linear(actual_dim, self.hidden_dim).to(self._device)
                    edge_attr_dict[edge_type] = self.edge_projs[edge_type_str](edge_store.edge_attr)

        # 多层异构消息传递
        for layer in self.layers:
            x_dict, edge_attr_dict = layer(x_dict, hetero_data.edge_index_dict, edge_attr_dict)


        all_nodes = torch.cat([x for x in x_dict.values()], dim=0)
        all_edges = torch.cat([x for x in edge_attr_dict.values()], dim=0)
        nodes_edges = torch.cat([all_nodes, all_edges], dim=0)

        graph_embedding = nodes_edges.mean(dim=0, keepdim=True)

        return graph_embedding


class HeteroMessagePassingLayer(nn.Module):
    """异构消息传递层 - 完整实现双消息传递机制"""

    def __init__(self, hidden_dim, n_heads=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

        assert hidden_dim % n_heads == 0, "hidden_dim must be divisible by n_heads"

        # 使用常规的字典来存储类型特定的参数，而不是ModuleDict
        self.node_attention_matrices = {}  # 存储节点注意力矩阵
        self.edge_attention_matrices = {}  # 存储边注意力矩阵
        self.node_type_matrices = {}  # 存储节点类型矩阵
        self.edge_type_matrices = {}  # 存储边类型矩阵

        # 输出投影
        self.W_O_node = nn.Linear(hidden_dim, hidden_dim)
        self.W_O_edge = nn.Linear(hidden_dim, hidden_dim)

        # 层归一化
        self.layer_norm_node1 = nn.LayerNorm(hidden_dim)
        self.layer_norm_node2 = nn.LayerNorm(hidden_dim)
        self.layer_norm_edge1 = nn.LayerNorm(hidden_dim)
        self.layer_norm_edge2 = nn.LayerNorm(hidden_dim)

        # 前馈网络
        self.ffn_node = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

        self.ffn_edge = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

        # 默认的注意力矩阵（用于未注册的类型）
        self.default_W_Q = nn.Linear(hidden_dim, hidden_dim)
        self.default_W_K = nn.Linear(hidden_dim, hidden_dim)
        self.default_W_V = nn.Linear(hidden_dim, hidden_dim)
        self.default_W_type = nn.Linear(hidden_dim, hidden_dim, bias=False)
        nn.init.eye_(self.default_W_type.weight)

        # 存储设备信息
        self._device = None

    def to(self, device):
        """重写to方法，确保所有组件都移动到设备"""
        super().to(device)
        self._device = device

        # 移动所有存储的矩阵到设备
        for key in list(self.node_attention_matrices.keys()):
            for matrix_type in ['Q', 'K', 'V']:
                if matrix_type in self.node_attention_matrices[key]:
                    self.node_attention_matrices[key][matrix_type] = self.node_attention_matrices[key][matrix_type].to(
                        device)

        for key in list(self.edge_attention_matrices.keys()):
            for matrix_type in ['Q', 'K', 'V']:
                if matrix_type in self.edge_attention_matrices[key]:
                    self.edge_attention_matrices[key][matrix_type] = self.edge_attention_matrices[key][matrix_type].to(
                        device)

        for key in list(self.node_type_matrices.keys()):
            self.node_type_matrices[key] = self.node_type_matrices[key].to(device)

        for key in list(self.edge_type_matrices.keys()):
            self.edge_type_matrices[key] = self.edge_type_matrices[key].to(device)

        return self

    def _init_attention_matrices(self, node_types, edge_types):
        """初始化注意力矩阵"""
        if self._device is None:
            self._device = next(self.parameters()).device

        # 初始化节点注意力矩阵
        for node_type in node_types:
            if node_type not in self.node_attention_matrices:
                self.node_attention_matrices[node_type] = {
                    'Q': nn.Linear(self.hidden_dim, self.hidden_dim).to(self._device),
                    'K': nn.Linear(self.hidden_dim, self.hidden_dim).to(self._device),
                    'V': nn.Linear(self.hidden_dim, self.hidden_dim).to(self._device)
                }

        # 初始化边注意力矩阵
        for edge_type in edge_types:
            edge_str = '__'.join(edge_type)
            if edge_str not in self.edge_attention_matrices:
                self.edge_attention_matrices[edge_str] = {
                    'Q': nn.Linear(self.hidden_dim, self.hidden_dim).to(self._device),
                    'K': nn.Linear(self.hidden_dim, self.hidden_dim).to(self._device),
                    'V': nn.Linear(self.hidden_dim, self.hidden_dim).to(self._device)
                }

    def _init_type_matrices(self, node_types, edge_types):
        """初始化类型特定的变换矩阵"""
        if self._device is None:
            self._device = next(self.parameters()).device

        for node_type in node_types:
            if node_type not in self.node_type_matrices:
                linear_layer = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
                nn.init.eye_(linear_layer.weight)
                self.node_type_matrices[node_type] = linear_layer.to(self._device)

        for edge_type in edge_types:
            edge_str = '__'.join(edge_type)
            if edge_str not in self.edge_type_matrices:
                linear_layer = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
                nn.init.eye_(linear_layer.weight)
                self.edge_type_matrices[edge_str] = linear_layer.to(self._device)

    def _get_node_attention_matrix(self, node_type, matrix_type):
        """获取节点注意力矩阵"""
        if node_type in self.node_attention_matrices and matrix_type in self.node_attention_matrices[node_type]:
            return self.node_attention_matrices[node_type][matrix_type]
        else:
            # 返回默认矩阵
            if matrix_type == 'Q':
                return self.default_W_Q
            elif matrix_type == 'K':
                return self.default_W_K
            elif matrix_type == 'V':
                return self.default_W_V
            else:
                raise ValueError(f"Unknown matrix type: {matrix_type}")

    def _get_edge_attention_matrix(self, edge_str, matrix_type):
        """获取边注意力矩阵"""
        if edge_str in self.edge_attention_matrices and matrix_type in self.edge_attention_matrices[edge_str]:
            return self.edge_attention_matrices[edge_str][matrix_type]
        else:
            # 返回默认矩阵
            if matrix_type == 'Q':
                return self.default_W_Q
            elif matrix_type == 'K':
                return self.default_W_K
            elif matrix_type == 'V':
                return self.default_W_V
            else:
                raise ValueError(f"Unknown matrix type: {matrix_type}")

    def _get_node_type_matrix(self, node_type):
        """获取节点类型矩阵"""
        if node_type in self.node_type_matrices:
            return self.node_type_matrices[node_type]
        else:
            return self.default_W_type

    def _get_edge_type_matrix(self, edge_str):
        """获取边类型矩阵"""
        if edge_str in self.edge_type_matrices:
            return self.edge_type_matrices[edge_str]
        else:
            return self.default_W_type

    def _multi_head_attention(self, Q, K, V, mask=None):
        """多头注意力机制 """
        batch_size, seq_len, _ = Q.size()

        # 线性投影并分头
        Q = Q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)  # [batch_size, n_heads, seq_len, head_dim]
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)

        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(
            self.head_dim)  # [batch_size, n_heads, seq_len, seq_len]

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # 注意力权重
        attention_weights = F.softmax(scores, dim=-1)

        # 应用注意力权重
        output = torch.matmul(attention_weights, V)  # [batch_size, n_heads, seq_len, head_dim]

        # 合并多头
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len,
                                                          self.hidden_dim)  # [batch_size, seq_len, hidden_dim]

        return output

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        """完整的双消息传递机制 - 同时更新节点和边"""
        if self._device is None:
            self._device = next(self.parameters()).device

        # 初始化注意力矩阵和类型矩阵
        self._init_attention_matrices(x_dict.keys(), edge_index_dict.keys())
        self._init_type_matrices(x_dict.keys(), edge_index_dict.keys())

        updated_x_dict = {}
        updated_edge_attr_dict = {}

        # 节点聚合
        for node_type in x_dict.keys():
            #print(f"处理节点类型: {node_type}")

            # 收集所有连接到该节点的消息
            all_messages = []
            message_indices = []  # 记录每个消息对应的目标节点索引

            for edge_type in edge_index_dict.keys():
                src_type, _, tgt_type = edge_type
                if tgt_type != node_type:
                    continue

                edge_index = edge_index_dict[edge_type]

                # 检查edge_index是否为空
                if edge_index is None or edge_index.numel() == 0:
                    print(f"跳过空的边类型: {edge_type}")
                    continue

                # 确保edge_index是2D张量
                if edge_index.dim() != 2 or edge_index.size(0) != 2:
                    print(f"跳过无效的边索引形状: {edge_type}, 形状: {edge_index.shape}")
                    continue

                src_features = x_dict[src_type]

                if src_features.device != self._device:
                    src_features = src_features.to(self._device)

                edge_attr = None
                if edge_type in edge_attr_dict:
                    edge_attr = edge_attr_dict[edge_type]
                    if edge_attr is not None and edge_attr.device != self._device:
                        edge_attr = edge_attr.to(self._device)

                # 处理该边类型的节点聚合
                row, col = edge_index  # row: 源节点, col: 目标节点

                for target_idx in range(x_dict[node_type].size(0)):
                    # 找到连接到目标节点的所有源节点
                    mask = col == target_idx
                    source_indices = row[mask]

                    if len(source_indices) == 0:
                        continue

                    # 获取源节点特征
                    source_features = src_features[source_indices]

                    # 应用源节点类型变换
                    W_src = self._get_node_type_matrix(src_type)
                    source_features_transformed = W_src(source_features)

                    # 如果有边特征，处理边特征
                    if edge_attr is not None:
                        edge_features = edge_attr[mask]
                        # 应用边类型变换
                        edge_str = '__'.join(edge_type)
                        W_edge = self._get_edge_type_matrix(edge_str)
                        edge_features_transformed = W_edge(edge_features)
                        # 将边特征合并到源节点特征中
                        source_features_combined = source_features_transformed + edge_features_transformed
                    else:
                        source_features_combined = source_features_transformed

                    all_messages.append(source_features_combined)
                    message_indices.append(target_idx)

            # 如果有消息，进行聚合
            if all_messages:
                current_features = x_dict[node_type]
                if current_features.device != self._device:
                    current_features = current_features.to(self._device)

                # 为每个目标节点聚合消息
                aggregated_features = torch.zeros_like(current_features)
                count = torch.zeros(current_features.size(0), device=self._device)

                for msg, target_idx in zip(all_messages, message_indices):
                    if msg.size(0) == 0:
                        continue

                    # 目标节点查询
                    target_feature = current_features[target_idx].unsqueeze(0)  # [1, hidden_dim]
                    W_tgt = self._get_node_type_matrix(node_type)
                    target_feature_transformed = W_tgt(target_feature)

                    # 准备多头注意力输入
                    Q = self._get_node_attention_matrix(node_type, 'Q')(target_feature_transformed).unsqueeze(
                        0)  # [1, 1, hidden_dim]
                    K = self._get_node_attention_matrix(edge_str, 'K')(msg).unsqueeze(0)  # [1, num_sources, hidden_dim]
                    V = self._get_node_attention_matrix(edge_str, 'V')(msg).unsqueeze(0)  # [1, num_sources, hidden_dim]

                    # 多头注意力
                    attended_msg = self._multi_head_attention(Q, K, V)  # [1, 1, hidden_dim]
                    attended_msg = self.W_O_node(attended_msg.squeeze(0))  # [1, hidden_dim]

                    aggregated_features[target_idx] += attended_msg.squeeze(0)
                    count[target_idx] += 1

                # 平均聚合
                mask = count > 0
                if mask.any():
                    aggregated_features[mask] = aggregated_features[mask] / count[mask].unsqueeze(1)

                # 节点更新
                x_updated = self.layer_norm_node1(current_features + aggregated_features)
                x_updated = self.layer_norm_node2(x_updated + self.ffn_node(x_updated))
                updated_x_dict[node_type] = x_updated
            else:
                updated_x_dict[node_type] = x_dict[node_type]

        # 边聚合
        for edge_type in edge_index_dict.keys():
            #print(f"处理边类型: {edge_type}")

            src_type, _, tgt_type = edge_type
            edge_index = edge_index_dict[edge_type]

            # 检查edge_index是否为空
            if edge_index is None or edge_index.numel() == 0:
                print(f"跳过空的边类型: {edge_type}")
                continue

            # 确保edge_index是2D张量
            if edge_index.dim() != 2 or edge_index.size(0) != 2:
                print(f"跳过无效的边索引形状: {edge_type}, 形状: {edge_index.shape}")
                continue

            if src_type not in x_dict:
                print(f"警告: 源节点类型 {src_type} 不在 x_dict 中")
                continue

            src_features = x_dict[src_type]
            if src_features.device != self._device:
                src_features = src_features.to(self._device)

            current_edge_attr = None
            if edge_type in edge_attr_dict:
                current_edge_attr = edge_attr_dict[edge_type]
                if current_edge_attr is not None and current_edge_attr.device != self._device:
                    current_edge_attr = current_edge_attr.to(self._device)

            # 边聚合
            if current_edge_attr is not None and current_edge_attr.numel() > 0:
                updated_edges = []
                row, col = edge_index
                edge_str = '__'.join(edge_type)

                for edge_idx in range(edge_index.size(1)):
                    src_idx = row[edge_idx].item()

                    # 找到连接到源节点的所有边（源节点的入边）
                    incoming_edges_mask = (col == src_idx)
                    incoming_edges_indices = torch.where(incoming_edges_mask)[0]

                    # 收集消息：源节点 + 源节点的入边
                    messages = [src_features[src_idx].unsqueeze(0)]  # 源节点特征

                    if incoming_edges_indices.numel() > 0:
                        # 获取源节点的入边特征
                        incoming_edge_features = current_edge_attr[incoming_edges_indices]
                        messages.append(incoming_edge_features)

                    # 合并所有消息
                    all_messages = torch.cat(messages, dim=0)  # [num_messages, hidden_dim]

                    # 边的查询
                    current_edge = current_edge_attr[edge_idx].unsqueeze(0)  # [1, hidden_dim]
                    W_edge = self._get_edge_type_matrix(edge_str)
                    current_edge_transformed = W_edge(current_edge)

                    # 准备多头注意力输入
                    Q = self._get_edge_attention_matrix(edge_str, 'Q')(current_edge_transformed).unsqueeze(
                        0)  # [1, 1, hidden_dim]
                    K = self._get_edge_attention_matrix(edge_str, 'K')(all_messages).unsqueeze(
                        0)  # [1, num_messages, hidden_dim]
                    V = self._get_edge_attention_matrix(edge_str, 'V')(all_messages).unsqueeze(
                        0)  # [1, num_messages, hidden_dim]

                    # 多头注意力
                    attended_features = self._multi_head_attention(Q, K, V)  # [1, 1, hidden_dim]
                    attended_features = self.W_O_edge(attended_features.squeeze(0))  # [1, hidden_dim]

                    updated_edges.append(attended_features.squeeze(0))

                if updated_edges:
                    updated_edge_attr = torch.stack(updated_edges)

                    # 边更新
                    edge_updated = self.layer_norm_edge1(current_edge_attr + updated_edge_attr)
                    edge_updated = self.layer_norm_edge2(edge_updated + self.ffn_edge(edge_updated))
                    updated_edge_attr_dict[edge_type] = edge_updated
                else:
                    updated_edge_attr_dict[edge_type] = current_edge_attr
            else:
                # 如果没有原始边特征，创建一个简单的边特征
                updated_edge_attr_dict[edge_type] = torch.ones(edge_index.size(1), self.hidden_dim, device=self._device)

        return updated_x_dict, updated_edge_attr_dict

class Multi_view_Heterogeneous_Encoder(nn.Module):

    def __init__(self, node_dims, edge_dims, hidden_dim=256, projection_dim=256):
        super().__init__()

        # 三个视图的编码器
        self.molecule_encoder = HeteroGNNEncoder(node_dims, edge_dims, hidden_dim)
        self.element_encoder = HeteroGNNEncoder(node_dims, edge_dims, hidden_dim)
        self.drug_encoder = HeteroGNNEncoder(node_dims, edge_dims, hidden_dim)

        # 投影头
        self.molecule_projector = Projector(hidden_dim, hidden_dim, projection_dim)
        self.element_projector = Projector(hidden_dim, hidden_dim, projection_dim)
        self.drug_projector = Projector(hidden_dim, hidden_dim, projection_dim)

        # 预测头
        self.property_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        self.ddi_predictor = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # 存储设备信息
        self._device = None

    def to(self, device):
        """重写to方法，确保所有组件都移动到设备"""
        super().to(device)
        self._device = device

        # 确保所有编码器也在设备上
        self.molecule_encoder = self.molecule_encoder.to(device)
        self.element_encoder = self.element_encoder.to(device)
        self.drug_encoder = self.drug_encoder.to(device)

        return self

    def forward(self, molecule_data, element_data, drug_data, mode='pretrain'):
        if mode == 'pretrain':
            # 预训练模式：返回三个视图的投影表示
            mol_embed = self.molecule_encoder(molecule_data, 'atom')
            elem_embed = self.element_encoder(element_data, 'atom')
            drug_embed = self.drug_encoder(drug_data, 'atom')

            z_mol = self.molecule_projector(mol_embed)
            z_elem = self.element_projector(elem_embed)
            z_drug = self.drug_projector(drug_embed)

            return z_mol, z_elem, z_drug

        # elif mode == 'test':
        #     # 分子性质预测
        #     embed = self.molecule_encoder(molecule_data, 'atom')
        #     return self.property_predictor(embed)


class Projector(nn.Module):
    """投影头"""

    def __init__(self, input_dim, hidden_dim=512, output_dim=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.network(x)


class ContrastiveLoss(nn.Module):
    """对比损失"""

    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        # z1, z2: [batch_size, projection_dim]
        batch_size = z1.size(0)

        # 归一化
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        # 相似度矩阵
        similarity_matrix = torch.matmul(z1, z2.T) / self.temperature

        # 正样本对是对角线
        positives = torch.diag(similarity_matrix).unsqueeze(1)

        # 负样本对是其他所有对
        negatives = similarity_matrix

        # 对比损失
        numerator = torch.exp(positives)
        denominator = torch.exp(negatives).sum(dim=1, keepdim=True)

        loss = -torch.log(numerator / denominator)
        return loss.mean()


class Trainer:
    """KCHML训练器"""

    def __init__(self, model, contrastive_loss_fn, device='cuda'):
        self.model = model.to(device)
        self.contrastive_loss_fn = contrastive_loss_fn
        self.device = device

        # 优化器
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=1e-4, weight_decay=1e-5
        )

    def train_contrastive(self, dataloader):
        """对比学习训练"""
        self.model.train()
        total_loss = 0

        for batch in dataloader:
            # 获取三个视图的数据
            molecule_data = batch['molecule']
            element_data = batch['element']
            drug_data = batch['drug']

            # 将数据移动到设备
            molecule_data = molecule_data.to(self.device)
            element_data = element_data.to(self.device)
            drug_data = drug_data.to(self.device)

            # 前向传播
            z_mol, z_elem, z_drug = self.model(
                molecule_data, element_data, drug_data, mode='pretrain'
            )

            # 计算对比损失
            loss_mol_elem = self.contrastive_loss_fn(z_mol, z_elem)
            loss_mol_drug = self.contrastive_loss_fn(z_mol, z_drug)
            loss_elem_drug = self.contrastive_loss_fn(z_elem, z_drug)

            total_loss = loss_mol_elem + loss_mol_drug + loss_elem_drug

            # 反向传播
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

        return total_loss.item()

    def train_property(self, dataloader):
        """分子性质预测训练"""
        self.model.train()
        total_loss = 0

        for batch in dataloader:
            molecule_data = batch['molecule'].to(self.device)
            targets = batch['target'].to(self.device)

            predictions = self.model(molecule_data, None, None, mode='test')
            loss = F.mse_loss(predictions.squeeze(), targets)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)


class HeteroMolecularDataset(Dataset):
    """异构分子数据集"""

    def __init__(self, excel_file, graph_builder, max_samples=1000):
        self.df = pd.read_excel(excel_file)
        self.graph_builder = graph_builder

        # 限制样本数量以加速训练
        if max_samples and len(self.df) > max_samples:
            self.df = self.df.sample(max_samples, random_state=42)

        # 预处理数据
        self.process_data()

    def process_data(self):
        """处理数据"""
        self.valid_indices = []
        self.graphs = {'molecule': [], 'element': [], 'drug': []}

        print("Processing molecular data...")
        for idx, row in self.df.iterrows():
            if idx % 100 == 0:
                print(f"Processed {idx}/{len(self.df)} molecules")

            smiles = row['isosmiles']

            try:
                # 构建三个视图
                mol_graph = self.graph_builder.build_molecule_view(smiles)
                elem_graph = self.graph_builder.build_element_view(smiles)
                drug_graph = self.graph_builder.build_drug_view(smiles)

                if mol_graph is not None and elem_graph is not None and drug_graph is not None:
                    self.valid_indices.append(idx)
                    self.graphs['molecule'].append(mol_graph)
                    self.graphs['element'].append(elem_graph)
                    self.graphs['drug'].append(drug_graph)
            except Exception as e:
                print(f"Error processing molecule {idx}: {e}")
                continue

        print(f"Successfully processed {len(self.valid_indices)} molecules")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        return {
            'molecule': self.graphs['molecule'][idx],
            'element': self.graphs['element'][idx],
            'drug': self.graphs['drug'][idx],
            'cid': self.df.iloc[actual_idx]['PUBCHEM_CID']
        }


def hetero_collate_fn(batch):
    """异构数据批处理函数"""
    # 对于异构数据，我们需要分别处理每个图
    # 这里简化处理：返回原始异构数据列表
    return {
        'molecule': [item['molecule'] for item in batch],
        'element': [item['element'] for item in batch],
        'drug': [item['drug'] for item in batch],
        'cids': [item['cid'] for item in batch]
    }


class SimpleHeteroWrapper:
    """简化异构数据包装器 - 用于批处理"""

    def __init__(self, hetero_data_list):
        self.hetero_data_list = hetero_data_list

        # 添加缺失的属性
        if hetero_data_list:
            # 使用第一个数据样本的属性
            self.node_types = hetero_data_list[0].node_types
            self.edge_types = hetero_data_list[0].edge_types
        else:
            self.node_types = []
            self.edge_types = []

    def to(self, device):
        """将数据移动到设备"""
        for data in self.hetero_data_list:
            for node_type in data.node_types:
                if hasattr(data[node_type], 'x') and data[node_type].x is not None:
                    data[node_type].x = data[node_type].x.to(device)

            for edge_type in data.edge_types:
                edge_store = data[edge_type]
                if hasattr(edge_store, 'edge_index') and edge_store.edge_index is not None:
                    edge_store.edge_index = edge_store.edge_index.to(device)
                if hasattr(edge_store, 'edge_attr') and edge_store.edge_attr is not None:
                    edge_store.edge_attr = edge_store.edge_attr.to(device)
        # 确保包装器本身知道设备信息
        self.device = device

        return self

    # 添加这些方法以支持属性访问
    def __getitem__(self, key):
        # 如果key是节点类型，返回第一个数据样本的对应节点
        if isinstance(key, str) and key in self.node_types:
            return self.hetero_data_list[0][key]
        # 如果key是边类型元组，返回第一个数据样本的对应边
        elif isinstance(key, tuple) and len(key) == 3 and key in self.edge_types:
            return self.hetero_data_list[0][key]
        else:
            raise KeyError(f"Key {key} not found")

    def __contains__(self, key):
        if isinstance(key, str):
            return key in self.node_types
        elif isinstance(key, tuple) and len(key) == 3:
            return key in self.edge_types
        return False

