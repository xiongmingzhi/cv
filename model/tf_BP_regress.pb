
E
PlaceholderPlaceholder*
dtype0*
shape:?????????
?
layer1/kernelConst*
dtype0*?
value?B?"x?t@??B??p">??@>?N??kT??^;־Z???n`H???=?Y?>?x?>??>`ýjR??4?=U.U>?(?Ȝ>~Dw>JE=?"?>?z ?e?>?>l??Vֽ???=????p???????
X
layer1/kernel/readIdentitylayer1/kernel*
T0* 
_class
loc:@layer1/kernel
L
layer1/biasConst*
dtype0*)
value B"q?>?4=????z?9?(?ν
R
layer1/bias/readIdentitylayer1/bias*
T0*
_class
loc:@layer1/bias
g
layer1/MatMulMatMulPlaceholderlayer1/kernel/read*
T0*
transpose_b( *
transpose_a( 
Z
layer1/BiasAddBiasAddlayer1/MatMullayer1/bias/read*
T0*
data_formatNHWC
,
layer1/ReluRelulayer1/BiasAdd*
T0
R
layer2/kernelConst*
dtype0*-
value$B""!}???#T?cX?dD? ???
X
layer2/kernel/readIdentitylayer2/kernel*
T0* 
_class
loc:@layer2/kernel
<
layer2/biasConst*
dtype0*
valueB*Ӧ?
R
layer2/bias/readIdentitylayer2/bias*
T0*
_class
loc:@layer2/bias
g
layer2/MatMulMatMullayer1/Relulayer2/kernel/read*
T0*
transpose_b( *
transpose_a( 
Z
layer2/BiasAddBiasAddlayer2/MatMullayer2/bias/read*
T0*
data_formatNHWC
2
layer2/SigmoidSigmoidlayer2/BiasAdd*
T0 