"""
ChromaDB 검증 및 테스트 스크립트
구축된 벡터 DB의 상태를 확인하고 다양한 쿼리로 테스트
"""

import os
from dotenv import load_dotenv
import chromadb
from openai import OpenAI

# 환경 변수 로드
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=OPENAI_API_KEY)


def get_embedding(text, model="text-embedding-3-small"):
    """텍스트 임베딩 생성"""
    text = text.replace("\n", " ")
    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding


def load_chromadb():
    """ChromaDB 로드"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    db_path = os.path.join(project_root, "data", "vectordb")
    
    if not os.path.exists(db_path):
        print(f"❌ ChromaDB를 찾을 수 없습니다: {db_path}")
        return None
    
    print(f"📂 DB 경로: {db_path}")
    
    chroma_client = chromadb.PersistentClient(path=db_path)
    
    try:
        collection = chroma_client.get_collection(name="youth_policies")
        return collection
    except Exception as e:
        print(f"❌ 컬렉션을 찾을 수 없습니다: {e}")
        return None


def check_db_stats(collection):
    """DB 통계 확인"""
    print("\n" + "=" * 70)
    print("📊 ChromaDB 통계")
    print("=" * 70)
    
    count = collection.count()
    print(f"✅ 저장된 정책 수: {count}개")
    
    # 샘플 데이터 확인
    sample = collection.peek(limit=3)
    
    print(f"\n📄 샘플 데이터 (3개):")
    print("-" * 70)
    
    for i, (id, doc, metadata) in enumerate(zip(sample['ids'], sample['documents'], sample['metadatas']), 1):
        print(f"\n[{i}] ID: {id}")
        print(f"    정책명: {metadata.get('정책명', 'N/A')}")
        print(f"    분야: {metadata.get('중분류', 'N/A')}")
        print(f"    담당: {metadata.get('주관기관명', 'N/A')}")
        print(f"    내용: {doc[:150]}...")
    
    return count


def test_search(collection, query, top_k=5):
    """검색 테스트"""
    print("\n" + "=" * 70)
    print("🔍 검색 테스트")
    print("=" * 70)
    print(f"질문: {query}")
    print(f"검색 결과 수: {top_k}개\n")
    
    # 쿼리 임베딩
    query_embedding = get_embedding(query)
    
    # 검색
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    
    if not results['documents'][0]:
        print("❌ 검색 결과가 없습니다.")
        return
    
    print(f"✅ {len(results['documents'][0])}개 결과 발견\n")
    
    for i, (doc, metadata, distance) in enumerate(zip(
        results['documents'][0], 
        results['metadatas'][0],
        results['distances'][0] if 'distances' in results else [0] * top_k
    ), 1):
        print(f"[{i}] {metadata.get('정책명', 'N/A')}")
        print(f"    📍 분야: {metadata.get('중분류', 'N/A')}")
        print(f"    🏢 담당: {metadata.get('주관기관명', 'N/A')}")
        print(f"    👤 연령: {metadata.get('지원최소연령', '0')}세 ~ {metadata.get('지원최대연령', '0')}세")
        print(f"    💰 지원금: {metadata.get('최소지원금액', '0')}원 ~ {metadata.get('최대지원금액', '0')}원")
        print(f"    📅 신청기간: {metadata.get('신청기간', 'N/A')}")
        print(f"    🔗 URL: {metadata.get('신청URL', 'N/A')}")
        print(f"    📏 유사도 거리: {distance:.4f}")
        print(f"    📝 내용: {doc[:150]}...")
        print()


def interactive_search(collection):
    """대화형 검색"""
    print("\n" + "=" * 70)
    print("💬 대화형 검색 모드 (종료: 'quit', 'q', 'exit')")
    print("=" * 70)
    
    while True:
        try:
            query = input("\n질문을 입력하세요: ").strip()
            
            if query.lower() in ['quit', 'q', 'exit', '종료']:
                print("검색을 종료합니다.")
                break
            
            if not query:
                continue
            
            test_search(collection, query, top_k=3)
            
        except KeyboardInterrupt:
            print("\n\n검색을 종료합니다.")
            break
        except Exception as e:
            print(f"❌ 오류 발생: {e}")


def main():
    print("=" * 70)
    print("ChromaDB 검증 및 테스트")
    print("=" * 70)
    
    # DB 로드
    collection = load_chromadb()
    
    if not collection:
        return
    
    # 1. DB 통계 확인
    count = check_db_stats(collection)
    
    if count == 0:
        print("\n❌ DB가 비어있습니다. build_vectordb.py를 먼저 실행하세요.")
        return
    
    # 2. 미리 정의된 테스트 쿼리들
    test_queries = [
        "취업 지원 프로그램이 있나요?",
        "창업 관련 정책을 알려주세요",
        "청년 주거 지원 정책은?",
        "해외 취업이나 인턴십 프로그램",
        "교육 바우처 지원"
    ]
    
    print("\n" + "=" * 70)
    print("🧪 자동 테스트 쿼리")
    print("=" * 70)
    
    for query in test_queries:
        test_search(collection, query, top_k=3)
        input("\n[Enter]를 눌러 다음 테스트로 진행...")
    
    # 3. 대화형 검색
    print("\n" + "=" * 70)
    response = input("대화형 검색을 시작하시겠습니까? (y/n): ").strip().lower()
    
    if response in ['y', 'yes', 'ㅛ']:
        interactive_search(collection)
    
    print("\n✅ 검증 완료!")


if __name__ == "__main__":
    main()
