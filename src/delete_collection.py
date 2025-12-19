from chromadb import PersistentClient
client = PersistentClient(path="./chroma_store")
try:
    client.delete_collection(name="facts")
    print("Deleted 'facts' collection to resolve embedding function conflict.")
except:
    print("No 'facts' collection found, proceeding with new creation.")