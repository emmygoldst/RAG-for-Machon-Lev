{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyMra190KrpuWW12q8mGX383",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/emmygoldst/RAG-for-Machon-Lev/blob/main/embedding%26faiss.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "Qmm4dB-_17UN"
      },
      "outputs": [],
      "source": [
        "from sentence_transformers import SentenceTransformer, util\n",
        "import faiss\n",
        "import subprocess  # Added for Linux commands\n",
        "\n",
        "class Embedder:\n",
        "    def __init__(self, model: str, content: list[dict]) -> None:\n",
        "        '''\n",
        "        Initializes the Embedder with a pre-trained model and content chunks.\n",
        "\n",
        "        Args:\n",
        "            model: Name/path of SentenceTransformer model (e.g., 'all-MiniLM-L6-v2')\n",
        "            content: List of dictionaries containing 'content' and metadata\n",
        "\n",
        "        Initializes:\n",
        "            chunks: Stores the input content\n",
        "            model: Loaded SentenceTransformer model\n",
        "            index: FAISS index built from content embeddings\n",
        "        '''\n",
        "        self.chunks = content\n",
        "        self.model = SentenceTransformer(model)\n",
        "        self.index = self.faiss_index(self.embedding())\n",
        "\n",
        "    def embedding(self) -> list:\n",
        "        '''\n",
        "        Generates embeddings for all content chunks.\n",
        "\n",
        "        Returns:\n",
        "            Numpy array of embeddings (shape: [num_chunks, embedding_dim])\n",
        "\n",
        "        Raises:\n",
        "            SystemExit: If embedding generation fails\n",
        "        '''\n",
        "        chunks_to_embed = [i['content'] for i in self.chunks]\n",
        "        try:\n",
        "            embeddings = self.model.encode(\n",
        "                chunks_to_embed,\n",
        "                normalize_embeddings=True,\n",
        "                show_progress_bar=True\n",
        "            )\n",
        "            return embeddings\n",
        "        except Exception as e:\n",
        "            print(f\"Error during embedding generation: {e}\")\n",
        "            exit(1)\n",
        "\n",
        "    def faiss_index(self, embeddings) -> faiss.Index:\n",
        "        '''\n",
        "        Creates a FAISS index from embeddings.\n",
        "\n",
        "        Args:\n",
        "            embeddings: Numpy array of embeddings\n",
        "\n",
        "        Returns:\n",
        "            FAISS index object\n",
        "\n",
        "        Raises:\n",
        "            SystemExit: If index creation fails\n",
        "        '''\n",
        "        try:\n",
        "            index = faiss.IndexFlatL2(embeddings.shape[1])\n",
        "            index.add(embeddings)\n",
        "            return index\n",
        "        except Exception as e:\n",
        "            print(f\"Error during FAISS index creation: {e}\")\n",
        "            exit(1)\n",
        "\n",
        "    def retrieve_top_k_chunks(self, query: str, k: int = 3) -> list:\n",
        "        '''\n",
        "        Retrieves top-k most relevant chunks for a query.\n",
        "\n",
        "        Args:\n",
        "            query: Search query string\n",
        "            k: Number of results to return (default: 3)\n",
        "\n",
        "        Returns:\n",
        "            List of top-k matching content strings\n",
        "        '''\n",
        "        chunk_texts = [chunk['content'] for chunk in self.chunks]\n",
        "        query_embedding = self.model.encode(query, convert_to_tensor=True)\n",
        "        query_embedding = query_embedding.cpu().numpy().astype('float32').reshape(1, -1)\n",
        "\n",
        "        distances, indices = self.index.search(query_embedding, k)\n",
        "        return [chunk_texts[i] for i in indices[0]]"
      ]
    }
  ]
}