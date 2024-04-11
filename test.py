import tensorflow as tf

# Example input IDs (replace with your actual input IDs)
input_ids = tf.constant([[-31, 51, 99], [15, 5, 0]])

# Create attention mask
attention_mask = tf.where(input_ids != 0, 1, 0)

print("Input IDs:", input_ids)
print("Attention Mask:", attention_mask)

