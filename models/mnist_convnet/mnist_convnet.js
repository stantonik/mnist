
const mnist_convnet = (() => {
const getTensorBuffer = (safetensorBuffer, tensorMetadata) => {
  return safetensorBuffer.subarray(...tensorMetadata.data_offsets);
};

const getTensorMetadata = (safetensorBuffer) => {
    const metadataLength = Number(new DataView(safetensorBuffer.buffer).getBigUint64(0, true));
    const metadata = JSON.parse(new TextDecoder("utf8").decode(safetensorBuffer.subarray(8, 8 + metadataLength)));
    return Object.fromEntries(Object.entries(metadata).filter(([k, v]) => k !== "__metadata__").map(([k, v]) => [k, {...v, data_offsets: v.data_offsets.map(x => 8 + metadataLength + x)}]));
};

const createEmptyBuf = (device, size) => {
    return device.createBuffer({size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
};

const createUniformBuf = (device, size) => {
  return device.createBuffer({size, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST})
}

const createInfinityUniformBuf = (device) => {
  const size = 4;
  const buf = device.createBuffer({
    mappedAtCreation: true,
    size,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
  });
  new Float32Array(buf.getMappedRange())[0] = Infinity;
  buf.unmap();
  return buf;
};

const createWeightBuf = (device, size, data) => {
  const buf = device.createBuffer({ size, usage: GPUBufferUsage.STORAGE, mappedAtCreation: true });
  new Uint8Array(buf.getMappedRange()).set(data); buf.unmap();
  return buf;
};

const addComputePass = (device, commandEncoder, pipeline, layout, infinityUniformBuf, bufs, workgroup) => {
  const bindGroup = device.createBindGroup({
    layout: layout,
    entries: [
      { binding: 0, resource: { buffer: infinityUniformBuf } },
      ...bufs.map((buffer, index) => ({ binding: index + 1, resource: { buffer } }))
    ]
  });

  const passEncoder = commandEncoder.beginComputePass();
  passEncoder.setPipeline(pipeline);
  passEncoder.setBindGroup(0, bindGroup);
  passEncoder.dispatchWorkgroups(...workgroup);
  passEncoder.end();
};

const r_13_13_8_2_2_4_3_3 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_21632:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_784:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_288:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_32:array<f32>;
@compute @workgroup_size(8,2,2) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 13 */
  var gidx1 = i32(gindex.y); /* 13 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 2 */
  var lidx2 = i32(lindex.z); /* 2 */
  var precast0 = gidx0;
  var precast1 = lidx0;
  var alu0 = (lidx0*36);
  var val0 = data2_288[alu0];
  var val1 = data2_288[(alu0+1)];
  var val2 = data2_288[(alu0+2)];
  var val3 = data2_288[(alu0+3)];
  var val4 = data2_288[(alu0+4)];
  var val5 = data2_288[(alu0+5)];
  var val6 = data2_288[(alu0+6)];
  var val7 = data2_288[(alu0+7)];
  var val8 = data2_288[(alu0+8)];
  var val9 = data2_288[(alu0+9)];
  var val10 = data2_288[(alu0+10)];
  var val11 = data2_288[(alu0+11)];
  var val12 = data2_288[(alu0+12)];
  var val13 = data2_288[(alu0+13)];
  var val14 = data2_288[(alu0+14)];
  var val15 = data2_288[(alu0+15)];
  var val16 = data2_288[(alu0+16)];
  var val17 = data2_288[(alu0+17)];
  var val18 = data2_288[(alu0+18)];
  var val19 = data2_288[(alu0+19)];
  var val20 = data2_288[(alu0+20)];
  var val21 = data2_288[(alu0+21)];
  var val22 = data2_288[(alu0+22)];
  var val23 = data2_288[(alu0+23)];
  var val24 = data2_288[(alu0+24)];
  var val25 = data2_288[(alu0+25)];
  var val26 = data2_288[(alu0+26)];
  var val27 = data2_288[(alu0+27)];
  var val28 = data2_288[(alu0+28)];
  var val29 = data2_288[(alu0+29)];
  var val30 = data2_288[(alu0+30)];
  var val31 = data2_288[(alu0+31)];
  var val32 = data2_288[(alu0+32)];
  var val33 = data2_288[(alu0+33)];
  var val34 = data2_288[(alu0+34)];
  var val35 = data2_288[(alu0+35)];
  var precast2 = (bitcast<u32>(precast0)<<1u);
  var cast0 = bitcast<i32>(precast2);
  var alu1 = (lidx2+cast0+(gidx1*56)+(lidx1*28));
  var val36 = data1_784[alu1];
  var val37 = data1_784[(alu1+1)];
  var val38 = data1_784[(alu1+2)];
  var val39 = data1_784[(alu1+28)];
  var val40 = data1_784[(alu1+29)];
  var val41 = data1_784[(alu1+30)];
  var val42 = data1_784[(alu1+56)];
  var val43 = data1_784[(alu1+57)];
  var val44 = data1_784[(alu1+58)];
  var precast3 = (bitcast<u32>(precast1)<<2u);
  var cast1 = bitcast<i32>(precast3);
  var val45 = data3_32[cast1];
  var val46 = data3_32[(cast1+1)];
  var val47 = data3_32[(cast1+2)];
  var val48 = data3_32[(cast1+3)];
  var alu2 = (lidx2+cast0+(gidx1*52)+(lidx0*2704)+(lidx1*26));
  data0_21632[alu2] = ((val36*val0)+(val39*val3)+(val42*val6)+(val37*val1)+(val40*val4)+(val43*val7)+(val38*val2)+(val41*val5)+(val44*val8)+val45);
  data0_21632[(alu2+676)] = ((val36*val9)+(val39*val12)+(val42*val15)+(val37*val10)+(val40*val13)+(val43*val16)+(val38*val11)+(val41*val14)+(val44*val17)+val46);
  data0_21632[(alu2+1352)] = ((val36*val18)+(val39*val21)+(val42*val24)+(val37*val19)+(val40*val22)+(val43*val25)+(val38*val20)+(val41*val23)+(val44*val26)+val47);
  data0_21632[(alu2+2028)] = ((val36*val27)+(val39*val30)+(val42*val33)+(val37*val28)+(val40*val31)+(val43*val34)+(val38*val29)+(val41*val32)+(val44*val35)+val48);
}`;

const r_13_13_8_4_2_2 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_5408:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_21632:array<f32>;
@compute @workgroup_size(8) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 13 */
  var gidx1 = i32(gindex.y); /* 13 */
  var lidx0 = i32(lindex.x); /* 8 */
  var precast0 = gidx0;
  var precast1 = (bitcast<u32>(precast0)<<1u);
  var alu0 = (bitcast<i32>(precast1)+(gidx1*1664)+(lidx0*208));
  var val0 = data1_21632[alu0];
  var val1 = data1_21632[(alu0+1)];
  var val2 = data1_21632[(alu0+26)];
  var val3 = data1_21632[(alu0+27)];
  var val4 = data1_21632[(alu0+52)];
  var val5 = data1_21632[(alu0+53)];
  var val6 = data1_21632[(alu0+78)];
  var val7 = data1_21632[(alu0+79)];
  var val8 = data1_21632[(alu0+104)];
  var val9 = data1_21632[(alu0+105)];
  var val10 = data1_21632[(alu0+130)];
  var val11 = data1_21632[(alu0+131)];
  var val12 = data1_21632[(alu0+156)];
  var val13 = data1_21632[(alu0+157)];
  var val14 = data1_21632[(alu0+182)];
  var val15 = data1_21632[(alu0+183)];
  var alu1 = (gidx0+(gidx1*416)+(lidx0*52));
  var alu2 = select(0.0f,val0,(0.0f<val0));
  var alu3 = select(0.0f,val1,(0.0f<val1));
  var alu4 = select(0.0f,val2,(0.0f<val2));
  var alu5 = select(alu2,alu4,(alu2<alu4));
  var alu6 = select(alu5,alu3,(alu5<alu3));
  var alu7 = select(0.0f,val3,(0.0f<val3));
  var alu8 = select(alu6,alu7,(alu6<alu7));
  data0_5408[alu1] = alu8;
  var alu10 = select(0.0f,val4,(0.0f<val4));
  var alu11 = select(0.0f,val5,(0.0f<val5));
  var alu12 = select(0.0f,val6,(0.0f<val6));
  var alu13 = select(alu10,alu12,(alu10<alu12));
  var alu14 = select(alu13,alu11,(alu13<alu11));
  var alu15 = select(0.0f,val7,(0.0f<val7));
  var alu16 = select(alu14,alu15,(alu14<alu15));
  data0_5408[(alu1+13)] = alu16;
  var alu18 = select(0.0f,val8,(0.0f<val8));
  var alu19 = select(0.0f,val9,(0.0f<val9));
  var alu20 = select(0.0f,val10,(0.0f<val10));
  var alu21 = select(alu18,alu20,(alu18<alu20));
  var alu22 = select(alu21,alu19,(alu21<alu19));
  var alu23 = select(0.0f,val11,(0.0f<val11));
  var alu24 = select(alu22,alu23,(alu22<alu23));
  data0_5408[(alu1+26)] = alu24;
  var alu26 = select(0.0f,val12,(0.0f<val12));
  var alu27 = select(0.0f,val13,(0.0f<val13));
  var alu28 = select(0.0f,val14,(0.0f<val14));
  var alu29 = select(alu26,alu28,(alu26<alu28));
  var alu30 = select(alu29,alu27,(alu29<alu27));
  var alu31 = select(0.0f,val15,(0.0f<val15));
  var alu32 = select(alu30,alu31,(alu30<alu31));
  data0_5408[(alu1+39)] = alu32;
}`;

const r_11_11_16_4_32_3_3 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_7744:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_5408:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_18432:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_64:array<f32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,4>;
  var gidx0 = i32(gindex.x); /* 11 */
  var gidx1 = i32(gindex.y); /* 11 */
  var lidx0 = i32(lindex.x); /* 16 */
  acc0[0] = 0.0f;
  acc0[1] = 0.0f;
  acc0[2] = 0.0f;
  acc0[3] = 0.0f;
  for (var ridx1004 = 0; ridx1004 < 32; ridx1004++) {
    var alu4 = ((lidx0*1152)+(ridx1004*9));
    var val0 = data2_18432[alu4];
    var val1 = data2_18432[(alu4+1)];
    var val2 = data2_18432[(alu4+2)];
    var val3 = data2_18432[(alu4+3)];
    var val4 = data2_18432[(alu4+4)];
    var val5 = data2_18432[(alu4+5)];
    var val6 = data2_18432[(alu4+6)];
    var val7 = data2_18432[(alu4+7)];
    var val8 = data2_18432[(alu4+8)];
    var val9 = data2_18432[(alu4+288)];
    var val10 = data2_18432[(alu4+289)];
    var val11 = data2_18432[(alu4+290)];
    var val12 = data2_18432[(alu4+291)];
    var val13 = data2_18432[(alu4+292)];
    var val14 = data2_18432[(alu4+293)];
    var val15 = data2_18432[(alu4+294)];
    var val16 = data2_18432[(alu4+295)];
    var val17 = data2_18432[(alu4+296)];
    var val18 = data2_18432[(alu4+576)];
    var val19 = data2_18432[(alu4+577)];
    var val20 = data2_18432[(alu4+578)];
    var val21 = data2_18432[(alu4+579)];
    var val22 = data2_18432[(alu4+580)];
    var val23 = data2_18432[(alu4+581)];
    var val24 = data2_18432[(alu4+582)];
    var val25 = data2_18432[(alu4+583)];
    var val26 = data2_18432[(alu4+584)];
    var val27 = data2_18432[(alu4+864)];
    var val28 = data2_18432[(alu4+865)];
    var val29 = data2_18432[(alu4+866)];
    var val30 = data2_18432[(alu4+867)];
    var val31 = data2_18432[(alu4+868)];
    var val32 = data2_18432[(alu4+869)];
    var val33 = data2_18432[(alu4+870)];
    var val34 = data2_18432[(alu4+871)];
    var val35 = data2_18432[(alu4+872)];
    var alu5 = (gidx0+(gidx1*13)+(ridx1004*169));
    var val36 = data1_5408[alu5];
    var val37 = data1_5408[(alu5+1)];
    var val38 = data1_5408[(alu5+2)];
    var val39 = data1_5408[(alu5+13)];
    var val40 = data1_5408[(alu5+14)];
    var val41 = data1_5408[(alu5+15)];
    var val42 = data1_5408[(alu5+26)];
    var val43 = data1_5408[(alu5+27)];
    var val44 = data1_5408[(alu5+28)];
    acc0[0] = (acc0[0]+(val36*val0)+(val39*val3)+(val42*val6)+(val37*val1)+(val40*val4)+(val43*val7)+(val38*val2)+(val41*val5)+(val44*val8));
    acc0[1] = (acc0[1]+(val36*val9)+(val39*val12)+(val42*val15)+(val37*val10)+(val40*val13)+(val43*val16)+(val38*val11)+(val41*val14)+(val44*val17));
    acc0[2] = (acc0[2]+(val36*val18)+(val39*val21)+(val42*val24)+(val37*val19)+(val40*val22)+(val43*val25)+(val38*val20)+(val41*val23)+(val44*val26));
    acc0[3] = (acc0[3]+(val36*val27)+(val39*val30)+(val42*val33)+(val37*val28)+(val40*val31)+(val43*val34)+(val38*val29)+(val41*val32)+(val44*val35));
  }
  var precast0 = lidx0;
  var precast1 = (bitcast<u32>(precast0)<<2u);
  var cast0 = bitcast<i32>(precast1);
  var val45 = data3_64[cast0];
  var val46 = data3_64[(cast0+1)];
  var val47 = data3_64[(cast0+2)];
  var val48 = data3_64[(cast0+3)];
  var alu11 = (gidx0+(gidx1*11)+(lidx0*484));
  data0_7744[alu11] = (acc0[0]+val45);
  data0_7744[(alu11+121)] = (acc0[1]+val46);
  data0_7744[(alu11+242)] = (acc0[2]+val47);
  data0_7744[(alu11+363)] = (acc0[3]+val48);
}`;

const r_5_5_16_4_2_2 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_1600:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_7744:array<f32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 5 */
  var gidx1 = i32(gindex.y); /* 5 */
  var lidx0 = i32(lindex.x); /* 16 */
  var precast0 = gidx0;
  var precast1 = (bitcast<u32>(precast0)<<1u);
  var alu0 = (bitcast<i32>(precast1)+(gidx1*22)+(lidx0*484));
  var val0 = data1_7744[alu0];
  var val1 = data1_7744[(alu0+1)];
  var val2 = data1_7744[(alu0+11)];
  var val3 = data1_7744[(alu0+12)];
  var val4 = data1_7744[(alu0+121)];
  var val5 = data1_7744[(alu0+122)];
  var val6 = data1_7744[(alu0+132)];
  var val7 = data1_7744[(alu0+133)];
  var val8 = data1_7744[(alu0+242)];
  var val9 = data1_7744[(alu0+243)];
  var val10 = data1_7744[(alu0+253)];
  var val11 = data1_7744[(alu0+254)];
  var val12 = data1_7744[(alu0+363)];
  var val13 = data1_7744[(alu0+364)];
  var val14 = data1_7744[(alu0+374)];
  var val15 = data1_7744[(alu0+375)];
  var alu1 = (gidx0+(gidx1*5)+(lidx0*100));
  var alu2 = select(0.0f,val0,(0.0f<val0));
  var alu3 = select(0.0f,val1,(0.0f<val1));
  var alu4 = select(0.0f,val2,(0.0f<val2));
  var alu5 = select(alu2,alu4,(alu2<alu4));
  var alu6 = select(alu5,alu3,(alu5<alu3));
  var alu7 = select(0.0f,val3,(0.0f<val3));
  var alu8 = select(alu6,alu7,(alu6<alu7));
  data0_1600[alu1] = alu8;
  var alu10 = select(0.0f,val4,(0.0f<val4));
  var alu11 = select(0.0f,val5,(0.0f<val5));
  var alu12 = select(0.0f,val6,(0.0f<val6));
  var alu13 = select(alu10,alu12,(alu10<alu12));
  var alu14 = select(alu13,alu11,(alu13<alu11));
  var alu15 = select(0.0f,val7,(0.0f<val7));
  var alu16 = select(alu14,alu15,(alu14<alu15));
  data0_1600[(alu1+25)] = alu16;
  var alu18 = select(0.0f,val8,(0.0f<val8));
  var alu19 = select(0.0f,val9,(0.0f<val9));
  var alu20 = select(0.0f,val10,(0.0f<val10));
  var alu21 = select(alu18,alu20,(alu18<alu20));
  var alu22 = select(alu21,alu19,(alu21<alu19));
  var alu23 = select(0.0f,val11,(0.0f<val11));
  var alu24 = select(alu22,alu23,(alu22<alu23));
  data0_1600[(alu1+50)] = alu24;
  var alu26 = select(0.0f,val12,(0.0f<val12));
  var alu27 = select(0.0f,val13,(0.0f<val13));
  var alu28 = select(0.0f,val14,(0.0f<val14));
  var alu29 = select(alu26,alu28,(alu26<alu28));
  var alu30 = select(alu29,alu27,(alu29<alu27));
  var alu31 = select(0.0f,val15,(0.0f<val15));
  var alu32 = select(alu30,alu31,(alu30<alu31));
  data0_1600[(alu1+75)] = alu32;
}`;

const r_10_16_100 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> temp0: array<f32,16>;
@group(0) @binding(1)var<storage,read_write>data0_10:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_1600:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_16000:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_10:array<f32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,1>;
  var acc1: array<f32,1>;
  var gidx0 = i32(gindex.x); /* 10 */
  var lidx0 = i32(lindex.x); /* 16 */
  acc1[0] = 0.0f;
  var val0 = data3_10[gidx0];
  var alu1 = (lidx0*100);
  acc0[0] = 0.0f;
  for (var ridx3002 = 0; ridx3002 < 100; ridx3002++) {
    var val1 = data2_16000[((gidx0*1600)+alu1+ridx3002)];
    var val2 = data1_1600[(alu1+ridx3002)];
    acc0[0] = (acc0[0]+(val2*val1));
  }
  temp0[lidx0] = acc0[0];
  workgroupBarrier();
  if (((bool(lidx0))!=true)) {
    for (var ridx1001 = 0; ridx1001 < 16; ridx1001++) {
      var val3 = temp0[ridx1001];
      acc1[0] = (acc1[0]+val3);
    }
    data0_10[gidx0] = (acc1[0]+val0);
  }
}`;

const setupNet = async (device, safetensor) => {
    const metadata = getTensorMetadata(safetensor);
    const infinityBuf = createInfinityUniformBuf(device);

    const layouts=[device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]})]

    const buf_0 = createEmptyBuf(device, 86528);;
    const input0 = createEmptyBuf(device, 3136);;
    const buf_1 = createWeightBuf(device, 1152, getTensorBuffer(safetensor, metadata['l1.weight']));
    const buf_2 = createWeightBuf(device, 128, getTensorBuffer(safetensor, metadata['l1.bias']));
    const buf_3 = createEmptyBuf(device, 21632);;
    const buf_4 = createEmptyBuf(device, 30976);;
    const buf_5 = createWeightBuf(device, 73728, getTensorBuffer(safetensor, metadata['l2.weight']));
    const buf_6 = createWeightBuf(device, 256, getTensorBuffer(safetensor, metadata['l2.bias']));
    const buf_7 = createEmptyBuf(device, 6400);;
    const output0 = createEmptyBuf(device, 40);;
    const buf_8 = createWeightBuf(device, 64000, getTensorBuffer(safetensor, metadata['l3.weight']));
    const buf_9 = createWeightBuf(device, 40, getTensorBuffer(safetensor, metadata['l3.bias']));

    const gpuWriteBuffer0 = device.createBuffer({size:input0.size, usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.MAP_WRITE });

    const gpuReadBuffer0 = device.createBuffer({size:output0.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

    const kernels = [r_13_13_8_2_2_4_3_3, r_13_13_8_4_2_2, r_11_11_16_4_32_3_3, r_5_5_16_4_2_2, r_10_16_100];
    const pipelines = await Promise.all(kernels.map(async (name, i) => {
      return await device.createComputePipelineAsync({
          layout: device.createPipelineLayout({
              bindGroupLayouts: [layouts[i]],
          }),
          compute: {
              module: device.createShaderModule({
                  code: name,
              }),
              entryPoint: "main",
          },
      });
  }))

    return async (_input0) => {
        const commandEncoder = device.createCommandEncoder();
        await gpuWriteBuffer0.mapAsync(GPUMapMode.WRITE);
        new Float32Array(gpuWriteBuffer0.getMappedRange()).set(_input0);
        gpuWriteBuffer0.unmap();
        commandEncoder.copyBufferToBuffer(gpuWriteBuffer0, 0, input0, 0, gpuWriteBuffer0.size);
        addComputePass(device, commandEncoder, pipelines[0], layouts[0], infinityBuf, [buf_0, input0, buf_1, buf_2], [13, 13, 1]);
        addComputePass(device, commandEncoder, pipelines[1], layouts[1], infinityBuf, [buf_3, buf_0], [13, 13, 1]);
        addComputePass(device, commandEncoder, pipelines[2], layouts[2], infinityBuf, [buf_4, buf_3, buf_5, buf_6], [11, 11, 1]);
        addComputePass(device, commandEncoder, pipelines[3], layouts[3], infinityBuf, [buf_7, buf_4], [5, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[4], layouts[4], infinityBuf, [output0, buf_7, buf_8, buf_9], [10, 1, 1]);
        commandEncoder.copyBufferToBuffer(output0, 0, gpuReadBuffer0, 0, output0.size);
        const gpuCommands = commandEncoder.finish();
        device.queue.submit([gpuCommands]);

        await gpuReadBuffer0.mapAsync(GPUMapMode.READ);
        const resultBuffer0 = new Float32Array(gpuReadBuffer0.size/4);
        resultBuffer0.set(new Float32Array(gpuReadBuffer0.getMappedRange()));
        gpuReadBuffer0.unmap();
        return [resultBuffer0];
    }
}
const load = async (device, weight_path) => { return await fetch(weight_path).then(x => x.arrayBuffer()).then(x => setupNet(device, new Uint8Array(x))); }
return { load, setupNet };
})();
export default mnist_convnet;
