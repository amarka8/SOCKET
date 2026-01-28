# SOCKET: Soft Collision Kernel EsTimator for Sparse Attention

SOCKET offers:

- A long-context inference-time algorithm for efficient key scoring using soft LSH.
- Traditional LSH is not suitable for ranking and scoring keys efficiently; soft LSH overcomes this limitation.
- 1.5x higher throughput compared to full attention, as measured with GPT-FAST.

To reproduce our results, follow the instructions in /SOCKET/REPRODUCE.txt and /SOCKET/GPT-FAST/REPRODUCE.txt.
