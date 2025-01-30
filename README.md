## This is a file collection of proto, pb2, client & server as well as model inference
***model weight and data are not included***

- **tensor_pb2.py** contains messages
- **tensor_pb2_grpc.py** contains both client-side and server-side servicer
- **vgg_pre.py** is the model operations' encapsulation
- Client_vgg_gpu_part 是客户端运行的程序，运行时会要求
  - 输入服务器IP，格式为IPV4
  - 输入模型执行方式，我设置成四位二进制代表所有执行方式，所以应输入0（0000）-15（1111）
- Server_vgg_gpu_part是服务器端运行的程序，挂着就行
- 所有程序必须放在./test_vgg文件夹下
