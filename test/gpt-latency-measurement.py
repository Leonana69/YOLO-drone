import sys, time
# import tiktoken
# sys.path.append("..")
# from controller.llm_wrapper import LLMWrapper

# llm = LLMWrapper()
# enc = tiktoken.encoding_for_model("gpt-4")

# def prompt_output_measure(length):
#     prompt = 'Please generate the exact same output as the following text: '
#     for i in range(length // 2):
#         prompt += str(i % 10) + " "
#     return prompt

# def prompt_input_measure(length):
#     suffix = "Please ignore all the above text and just generate True"
#     prompt = ''
#     init_len = enc.encode(suffix)
#     for i in range((length - len(init_len)) // 2):
#         prompt += str(i % 10) + " "
#     return prompt + suffix

lengths = [8000]
# lengths = [50, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]
# lengths = [50, 100, 200, 300, 400]
result = []
# for length in lengths:
#     t = 0
#     input_length = 0
#     output_length = 0
#     for i in range(10):
#         # prompt = prompt_output_measure(length)
#         prompt = prompt_input_measure(length)
#         start = time.time()
#         input_length += len(enc.encode(prompt))
#         output = llm.request(prompt)
#         output_length += len(enc.encode(output))
#         t += time.time() - start
#         print(f"t: {t}, i: {input_length}, o: {output_length}")
#     print("Time taken for length", length, ":", t / 10)
#     print("Input length:", input_length / 10)
#     print("Output length:", output_length / 10)
#     result.append((length, t / 10, input_length / 10, output_length / 10))
# print(result)
# exit(0)

# different output
data_1 = [
    (50, 2.276, 62, 49),
    (100, 4.560, 112, 99),
    (200, 8.473, 212, 199),
    (300, 10.996, 312, 295),
    (400, 14.425, 412, 413),
]

# different input
# data_2 = [
#     (50, 0.47491774559020994, 49.0, 1.0), 
#     (100, 0.4766784429550171, 99.0, 1.0),
#     (200, 0.46629860401153567, 199.0, 1.0),
#     (300, 0.4480326175689697, 299.0, 1.0),
#     (400, 0.5139770269393921, 399.0, 1.0),
#     (1000, 0.4809334516525269, 999.0, 1.0),
#     (2000, 0.6343598604202271, 1999.0, 1.0),
#     (4000, 0.7674200057983398, 3999.0, 1.0),
#     (8000, 1.3128541946411132, 7999.0, 1.0)]
data_2 = [(50, 0.5378251791000366, 49.0, 1.0),
(500, 0.5108302307128907, 499.0, 1.0),
(1000, 0.4951801300048828, 999.0, 1.0),
(2000, 0.5111032485961914, 1999.0, 1.0),
(3000, 0.5264493227005005, 2999.0, 1.0),
(4000, 0.5382437705993652, 3999.0, 1.0),
(5000, 0.5212562322616577, 4999.0, 1.0),
(6000, 0.5919422626495361, 5999.0, 1.0),
(7000, 0.5916801214218139, 6999.0, 1.0),
(8000, 0.6088189125061035, 7999.0, 1.0)]

network_latency = 28.127

red_color = '#FF6B6B'
blue_color = '#4D96FF'
white_color = '#FFFFFF'
black_color = '#000000'

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

col1_1 = [x[0] for x in data_2]
col2 = [x[1] for x in data_2]

col1_2 = [x[0] for x in data_1]
col3 = [x[1] for x in data_1]

# Perform linear regression for each dataset
slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(col1_1, col2)

for i in range(len(data_1)):
    col3[i] -= data_1[i][2] * slope2

slope3, intercept3, r_value3, p_value3, std_err3 = stats.linregress(col1_2, col3)

# Create arrays from the x-coordinates for line plots
line_x1 = np.linspace(min(col1_1), max(col1_1), 100)  # For smoother line plot
line_x2 = np.linspace(min(col1_2), max(col1_2), 100)

# Create line equations for the plots
line2 = slope2 * line_x1 + intercept2
line3 = slope3 * line_x2 + intercept3

plt.rcParams.update({'legend.fontsize': 19, 'axes.edgecolor': 'black',
                     'axes.linewidth': 2.2, 'font.size': 25})

### plot in a single figure
# fig, ax1 = plt.subplots(figsize=[16, 6])
# plt.tight_layout(pad=2)
# # Plot the first dataset with its regression
# ax1.scatter(col1_1, col2, color=black_color, label='Various input, fixed output', marker='x', linewidth=3, s=200)
# ax1.plot(np.array(col1_1), slope2 * np.array(col1_1) + intercept2, '-', color=black_color, label=f'a={slope2:.6f}, r={r_value2:.4}', linewidth=3)
# ax1.set_xlabel('Input Token Number', color=black_color)
# ax1.set_ylabel('Time Taken (s)')
# ax1.tick_params(axis='x', labelcolor=black_color)
# ax1.tick_params(axis='y')

# # Create a second x-axis for the second dataset
# ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
# ax2.scatter(col1_2, col3, color=black_color, label='Fixed input, various output', linewidth=3, s=200)
# ax2.plot(np.array(col1_2), slope3 * np.array(col1_2) + intercept3, '--', color=black_color, label=f'b={slope3:.6f}, r={r_value3:.4}', linewidth=3)
# ax2.set_xlabel('Output Token Number', color=black_color)
# ax2.tick_params(axis='x', labelcolor=black_color)

# # Add legends and show plot
# ax1.legend(loc='lower right', bbox_to_anchor=(1, 0.12))
# ax2.legend(loc='upper left')
# # plt.show()

# plt.savefig('gpt4-latency.pdf')

### plot in two figures
fig, ax1 = plt.subplots(figsize=[14, 6])
plt.tight_layout(pad=2)
# Plot the first dataset with its regression
ax1.scatter(col1_1, col2, color=black_color, label='Various input, fixed output', marker='x', linewidth=3, s=200)
ax1.plot(np.array(col1_1), slope2 * np.array(col1_1) + intercept2, '-', color=black_color, label=f'a={slope2:.6f}, r={r_value2:.4}', linewidth=3)
ax1.set_xlabel('Input Token Number', color=black_color)
ax1.set_ylabel('Time Taken (s)')
ax1.tick_params(axis='x', labelcolor=black_color)
ax1.tick_params(axis='y')

# Add legends and show plot
ax1.legend(loc='upper left')
plt.savefig('gpt4-latency-input.pdf')
# plt.show()

fig, ax2 = plt.subplots(figsize=[14, 6])
plt.tight_layout(pad=2)
# Create a second x-axis for the second dataset
ax2.scatter(col1_2, col3, color=black_color, label='Fixed input, various output', linewidth=3, s=200)
ax2.plot(np.array(col1_2), slope3 * np.array(col1_2) + intercept3, '--', color=black_color, label=f'b={slope3:.6f}, r={r_value3:.4}', linewidth=3)
ax2.set_xlabel('Output Token Number', color=black_color)
ax2.set_ylabel('Time Taken (s)')
ax2.tick_params(axis='x', labelcolor=black_color)
ax2.tick_params(axis='y')

# Add legends and show plot
ax2.legend(loc='upper left')
plt.savefig('gpt4-latency-output.pdf')
# plt.show()
