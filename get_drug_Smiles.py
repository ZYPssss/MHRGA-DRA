import requests


def get_isomeric_smiles(self, cid: str) -> Optional[str]:
    """
    根据CID获取异构SMILES表示（包含立体化学信息）

    Args:
        cid: PubChem CID

    Returns:
        异构SMILES字符串，如果获取失败返回None
    """
    try:
        url = f"{self.base_url}/compound/cid/{cid}/property/IsomericSMILES/JSON"

        response = requests.get(url, timeout=10)
        print(f"CID {cid} 请求状态码: {response.status_code}")  # 调试信息

        if response.status_code == 200:
            data = response.json()
            # 检查Properties是否存在且非空
            if 'PropertyTable' in data and 'Properties' in data['PropertyTable'] and data['PropertyTable'][
                'Properties']:
                smiles = data['PropertyTable']['Properties'][0].get('IsomericSMILES')
                return smiles
            else:
                print(f"CID {cid}: 返回数据中未找到Properties")
                return None
        else:
            print(f"CID {cid}: 请求失败，状态码 {response.status_code}")
            # 打印返回的文本以获取错误信息
            print(f"响应文本: {response.text}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"CID {cid}: 网络请求错误 - {e}")
        return None
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        print(f"CID {cid}: 数据解析错误 - {e}")
        return None

# 使用示例
if __name__ == "__main__":
    test_cids = ["8378", "5287702", "441203"]  # 测试几个CID

    for cid in test_cids:
        smiles = get_pubchem_smiles(cid)
        print(f"CID {cid}: {smiles}")