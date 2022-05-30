
matmul:     file format elf64-x86-64


Disassembly of section .init:

0000000000400a38 <_init>:
  400a38:	48 83 ec 08          	sub    $0x8,%rsp
  400a3c:	48 8b 05 b5 35 20 00 	mov    0x2035b5(%rip),%rax        # 603ff8 <_DYNAMIC+0x220>
  400a43:	48 85 c0             	test   %rax,%rax
  400a46:	74 05                	je     400a4d <_init+0x15>
  400a48:	e8 53 01 00 00       	callq  400ba0 <__gmon_start__@plt>
  400a4d:	48 83 c4 08          	add    $0x8,%rsp
  400a51:	c3                   	retq   

Disassembly of section .plt:

0000000000400a60 <printf@plt-0x10>:
  400a60:	ff 35 a2 35 20 00    	pushq  0x2035a2(%rip)        # 604008 <_GLOBAL_OFFSET_TABLE_+0x8>
  400a66:	ff 25 a4 35 20 00    	jmpq   *0x2035a4(%rip)        # 604010 <_GLOBAL_OFFSET_TABLE_+0x10>
  400a6c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400a70 <printf@plt>:
  400a70:	ff 25 a2 35 20 00    	jmpq   *0x2035a2(%rip)        # 604018 <_GLOBAL_OFFSET_TABLE_+0x18>
  400a76:	68 00 00 00 00       	pushq  $0x0
  400a7b:	e9 e0 ff ff ff       	jmpq   400a60 <_init+0x28>

0000000000400a80 <sprintf@plt>:
  400a80:	ff 25 9a 35 20 00    	jmpq   *0x20359a(%rip)        # 604020 <_GLOBAL_OFFSET_TABLE_+0x20>
  400a86:	68 01 00 00 00       	pushq  $0x1
  400a8b:	e9 d0 ff ff ff       	jmpq   400a60 <_init+0x28>

0000000000400a90 <strlen@plt>:
  400a90:	ff 25 92 35 20 00    	jmpq   *0x203592(%rip)        # 604028 <_GLOBAL_OFFSET_TABLE_+0x28>
  400a96:	68 02 00 00 00       	pushq  $0x2
  400a9b:	e9 c0 ff ff ff       	jmpq   400a60 <_init+0x28>

0000000000400aa0 <memset@plt>:
  400aa0:	ff 25 8a 35 20 00    	jmpq   *0x20358a(%rip)        # 604030 <_GLOBAL_OFFSET_TABLE_+0x30>
  400aa6:	68 03 00 00 00       	pushq  $0x3
  400aab:	e9 b0 ff ff ff       	jmpq   400a60 <_init+0x28>

0000000000400ab0 <calloc@plt>:
  400ab0:	ff 25 82 35 20 00    	jmpq   *0x203582(%rip)        # 604038 <_GLOBAL_OFFSET_TABLE_+0x38>
  400ab6:	68 04 00 00 00       	pushq  $0x4
  400abb:	e9 a0 ff ff ff       	jmpq   400a60 <_init+0x28>

0000000000400ac0 <sscanf@plt>:
  400ac0:	ff 25 7a 35 20 00    	jmpq   *0x20357a(%rip)        # 604040 <_GLOBAL_OFFSET_TABLE_+0x40>
  400ac6:	68 05 00 00 00       	pushq  $0x5
  400acb:	e9 90 ff ff ff       	jmpq   400a60 <_init+0x28>

0000000000400ad0 <memcpy@plt>:
  400ad0:	ff 25 72 35 20 00    	jmpq   *0x203572(%rip)        # 604048 <_GLOBAL_OFFSET_TABLE_+0x48>
  400ad6:	68 06 00 00 00       	pushq  $0x6
  400adb:	e9 80 ff ff ff       	jmpq   400a60 <_init+0x28>

0000000000400ae0 <fclose@plt>:
  400ae0:	ff 25 6a 35 20 00    	jmpq   *0x20356a(%rip)        # 604050 <_GLOBAL_OFFSET_TABLE_+0x50>
  400ae6:	68 07 00 00 00       	pushq  $0x7
  400aeb:	e9 70 ff ff ff       	jmpq   400a60 <_init+0x28>

0000000000400af0 <fopen@plt>:
  400af0:	ff 25 62 35 20 00    	jmpq   *0x203562(%rip)        # 604058 <_GLOBAL_OFFSET_TABLE_+0x58>
  400af6:	68 08 00 00 00       	pushq  $0x8
  400afb:	e9 60 ff ff ff       	jmpq   400a60 <_init+0x28>

0000000000400b00 <free@plt>:
  400b00:	ff 25 5a 35 20 00    	jmpq   *0x20355a(%rip)        # 604060 <_GLOBAL_OFFSET_TABLE_+0x60>
  400b06:	68 09 00 00 00       	pushq  $0x9
  400b0b:	e9 50 ff ff ff       	jmpq   400a60 <_init+0x28>

0000000000400b10 <exit@plt>:
  400b10:	ff 25 52 35 20 00    	jmpq   *0x203552(%rip)        # 604068 <_GLOBAL_OFFSET_TABLE_+0x68>
  400b16:	68 0a 00 00 00       	pushq  $0xa
  400b1b:	e9 40 ff ff ff       	jmpq   400a60 <_init+0x28>

0000000000400b20 <malloc@plt>:
  400b20:	ff 25 4a 35 20 00    	jmpq   *0x20354a(%rip)        # 604070 <_GLOBAL_OFFSET_TABLE_+0x70>
  400b26:	68 0b 00 00 00       	pushq  $0xb
  400b2b:	e9 30 ff ff ff       	jmpq   400a60 <_init+0x28>

0000000000400b30 <strcmp@plt>:
  400b30:	ff 25 42 35 20 00    	jmpq   *0x203542(%rip)        # 604078 <_GLOBAL_OFFSET_TABLE_+0x78>
  400b36:	68 0c 00 00 00       	pushq  $0xc
  400b3b:	e9 20 ff ff ff       	jmpq   400a60 <_init+0x28>

0000000000400b40 <fprintf@plt>:
  400b40:	ff 25 3a 35 20 00    	jmpq   *0x20353a(%rip)        # 604080 <_GLOBAL_OFFSET_TABLE_+0x80>
  400b46:	68 0d 00 00 00       	pushq  $0xd
  400b4b:	e9 10 ff ff ff       	jmpq   400a60 <_init+0x28>

0000000000400b50 <tolower@plt>:
  400b50:	ff 25 32 35 20 00    	jmpq   *0x203532(%rip)        # 604088 <_GLOBAL_OFFSET_TABLE_+0x88>
  400b56:	68 0e 00 00 00       	pushq  $0xe
  400b5b:	e9 00 ff ff ff       	jmpq   400a60 <_init+0x28>

0000000000400b60 <puts@plt>:
  400b60:	ff 25 2a 35 20 00    	jmpq   *0x20352a(%rip)        # 604090 <_GLOBAL_OFFSET_TABLE_+0x90>
  400b66:	68 0f 00 00 00       	pushq  $0xf
  400b6b:	e9 f0 fe ff ff       	jmpq   400a60 <_init+0x28>

0000000000400b70 <fgets@plt>:
  400b70:	ff 25 22 35 20 00    	jmpq   *0x203522(%rip)        # 604098 <_GLOBAL_OFFSET_TABLE_+0x98>
  400b76:	68 10 00 00 00       	pushq  $0x10
  400b7b:	e9 e0 fe ff ff       	jmpq   400a60 <_init+0x28>

0000000000400b80 <gettimeofday@plt>:
  400b80:	ff 25 1a 35 20 00    	jmpq   *0x20351a(%rip)        # 6040a0 <_GLOBAL_OFFSET_TABLE_+0xa0>
  400b86:	68 11 00 00 00       	pushq  $0x11
  400b8b:	e9 d0 fe ff ff       	jmpq   400a60 <_init+0x28>

0000000000400b90 <__libc_start_main@plt>:
  400b90:	ff 25 12 35 20 00    	jmpq   *0x203512(%rip)        # 6040a8 <_GLOBAL_OFFSET_TABLE_+0xa8>
  400b96:	68 12 00 00 00       	pushq  $0x12
  400b9b:	e9 c0 fe ff ff       	jmpq   400a60 <_init+0x28>

0000000000400ba0 <__gmon_start__@plt>:
  400ba0:	ff 25 0a 35 20 00    	jmpq   *0x20350a(%rip)        # 6040b0 <_GLOBAL_OFFSET_TABLE_+0xb0>
  400ba6:	68 13 00 00 00       	pushq  $0x13
  400bab:	e9 b0 fe ff ff       	jmpq   400a60 <_init+0x28>

0000000000400bb0 <fscanf@plt>:
  400bb0:	ff 25 02 35 20 00    	jmpq   *0x203502(%rip)        # 6040b8 <_GLOBAL_OFFSET_TABLE_+0xb8>
  400bb6:	68 14 00 00 00       	pushq  $0x14
  400bbb:	e9 a0 fe ff ff       	jmpq   400a60 <_init+0x28>

0000000000400bc0 <fwrite@plt>:
  400bc0:	ff 25 fa 34 20 00    	jmpq   *0x2034fa(%rip)        # 6040c0 <_GLOBAL_OFFSET_TABLE_+0xc0>
  400bc6:	68 15 00 00 00       	pushq  $0x15
  400bcb:	e9 90 fe ff ff       	jmpq   400a60 <_init+0x28>

Disassembly of section .text:

0000000000400bd0 <main>:
  400bd0:	41 57                	push   %r15
  400bd2:	41 56                	push   %r14
  400bd4:	41 55                	push   %r13
  400bd6:	41 54                	push   %r12
  400bd8:	55                   	push   %rbp
  400bd9:	53                   	push   %rbx
  400bda:	48 81 ec c8 00 00 00 	sub    $0xc8,%rsp
  400be1:	83 ff 02             	cmp    $0x2,%edi
  400be4:	48 89 74 24 68       	mov    %rsi,0x68(%rsp)
  400be9:	c7 84 24 98 00 00 00 	movl   $0x0,0x98(%rsp)
  400bf0:	00 00 00 00 
  400bf4:	c7 84 24 9c 00 00 00 	movl   $0x0,0x9c(%rsp)
  400bfb:	00 00 00 00 
  400bff:	0f 8e d4 05 00 00    	jle    4011d9 <main+0x609>
  400c05:	48 8b 7e 08          	mov    0x8(%rsi),%rdi
  400c09:	be f0 2d 40 00       	mov    $0x402df0,%esi
  400c0e:	e8 dd fe ff ff       	callq  400af0 <fopen@plt>
  400c13:	48 85 c0             	test   %rax,%rax
  400c16:	48 89 c5             	mov    %rax,%rbp
  400c19:	0f 84 a6 05 00 00    	je     4011c5 <main+0x5f5>
  400c1f:	48 8b 44 24 68       	mov    0x68(%rsp),%rax
  400c24:	be f0 2d 40 00       	mov    $0x402df0,%esi
  400c29:	48 8b 78 10          	mov    0x10(%rax),%rdi
  400c2d:	e8 be fe ff ff       	callq  400af0 <fopen@plt>
  400c32:	48 85 c0             	test   %rax,%rax
  400c35:	48 89 c3             	mov    %rax,%rbx
  400c38:	0f 84 91 05 00 00    	je     4011cf <main+0x5ff>
  400c3e:	48 83 ec 08          	sub    $0x8,%rsp
  400c42:	49 89 e9             	mov    %rbp,%r9
  400c45:	50                   	push   %rax
  400c46:	4c 8d 84 24 ac 00 00 	lea    0xac(%rsp),%r8
  400c4d:	00 
  400c4e:	48 8d 8c 24 a8 00 00 	lea    0xa8(%rsp),%rcx
  400c55:	00 
  400c56:	48 8d 94 24 a4 00 00 	lea    0xa4(%rsp),%rdx
  400c5d:	00 
  400c5e:	48 8d b4 24 a0 00 00 	lea    0xa0(%rsp),%rsi
  400c65:	00 
  400c66:	48 8d bc 24 9c 00 00 	lea    0x9c(%rsp),%rdi
  400c6d:	00 
  400c6e:	e8 ad 0e 00 00       	callq  401b20 <_Z8read_matPiS_S_S_S_P8_IO_FILES1_>
  400c73:	83 f8 f1             	cmp    $0xfffffff1,%eax
  400c76:	5e                   	pop    %rsi
  400c77:	5f                   	pop    %rdi
  400c78:	0f 84 2d 05 00 00    	je     4011ab <main+0x5db>
  400c7e:	8b bc 24 8c 00 00 00 	mov    0x8c(%rsp),%edi
  400c85:	be 04 00 00 00       	mov    $0x4,%esi
  400c8a:	0f af bc 24 90 00 00 	imul   0x90(%rsp),%edi
  400c91:	00 
  400c92:	48 63 ff             	movslq %edi,%rdi
  400c95:	e8 16 fe ff ff       	callq  400ab0 <calloc@plt>
  400c9a:	48 85 c0             	test   %rax,%rax
  400c9d:	48 89 44 24 58       	mov    %rax,0x58(%rsp)
  400ca2:	0f 84 65 05 00 00    	je     40120d <main+0x63d>
  400ca8:	8b bc 24 90 00 00 00 	mov    0x90(%rsp),%edi
  400caf:	be 04 00 00 00       	mov    $0x4,%esi
  400cb4:	0f af bc 24 94 00 00 	imul   0x94(%rsp),%edi
  400cbb:	00 
  400cbc:	48 63 ff             	movslq %edi,%rdi
  400cbf:	e8 ec fd ff ff       	callq  400ab0 <calloc@plt>
  400cc4:	48 85 c0             	test   %rax,%rax
  400cc7:	49 89 c4             	mov    %rax,%r12
  400cca:	0f 84 29 05 00 00    	je     4011f9 <main+0x629>
  400cd0:	8b 8c 24 98 00 00 00 	mov    0x98(%rsp),%ecx
  400cd7:	85 c9                	test   %ecx,%ecx
  400cd9:	0f 8e 42 05 00 00    	jle    401221 <main+0x651>
  400cdf:	4c 8b 44 24 58       	mov    0x58(%rsp),%r8
  400ce4:	8b 94 24 90 00 00 00 	mov    0x90(%rsp),%edx
  400ceb:	48 89 ef             	mov    %rbp,%rdi
  400cee:	8b b4 24 8c 00 00 00 	mov    0x8c(%rsp),%esi
  400cf5:	e8 76 09 00 00       	callq  401670 <_Z11read_sparseP8_IO_FILEiiiPf>
  400cfa:	8b 8c 24 9c 00 00 00 	mov    0x9c(%rsp),%ecx
  400d01:	85 c9                	test   %ecx,%ecx
  400d03:	0f 8e 84 04 00 00    	jle    40118d <main+0x5bd>
  400d09:	8b 94 24 94 00 00 00 	mov    0x94(%rsp),%edx
  400d10:	8b b4 24 90 00 00 00 	mov    0x90(%rsp),%esi
  400d17:	4d 89 e0             	mov    %r12,%r8
  400d1a:	48 89 df             	mov    %rbx,%rdi
  400d1d:	e8 4e 09 00 00       	callq  401670 <_Z11read_sparseP8_IO_FILEiiiPf>
  400d22:	48 89 ef             	mov    %rbp,%rdi
  400d25:	e8 b6 fd ff ff       	callq  400ae0 <fclose@plt>
  400d2a:	48 89 df             	mov    %rbx,%rdi
  400d2d:	e8 ae fd ff ff       	callq  400ae0 <fclose@plt>
  400d32:	8b bc 24 8c 00 00 00 	mov    0x8c(%rsp),%edi
  400d39:	be 04 00 00 00       	mov    $0x4,%esi
  400d3e:	0f af bc 24 94 00 00 	imul   0x94(%rsp),%edi
  400d45:	00 
  400d46:	48 63 ff             	movslq %edi,%rdi
  400d49:	e8 62 fd ff ff       	callq  400ab0 <calloc@plt>
  400d4e:	48 85 c0             	test   %rax,%rax
  400d51:	49 89 c7             	mov    %rax,%r15
  400d54:	0f 84 1f 04 00 00    	je     401179 <main+0x5a9>
  400d5a:	8b bc 24 8c 00 00 00 	mov    0x8c(%rsp),%edi
  400d61:	be 04 00 00 00       	mov    $0x4,%esi
  400d66:	0f af bc 24 94 00 00 	imul   0x94(%rsp),%edi
  400d6d:	00 
  400d6e:	48 63 ff             	movslq %edi,%rdi
  400d71:	e8 3a fd ff ff       	callq  400ab0 <calloc@plt>
  400d76:	48 85 c0             	test   %rax,%rax
  400d79:	0f 84 fa 03 00 00    	je     401179 <main+0x5a9>
  400d7f:	48 8d bc 24 a0 00 00 	lea    0xa0(%rsp),%rdi
  400d86:	00 
  400d87:	31 f6                	xor    %esi,%esi
  400d89:	e8 f2 fd ff ff       	callq  400b80 <gettimeofday@plt>
  400d8e:	8b 84 24 90 00 00 00 	mov    0x90(%rsp),%eax
  400d95:	44 8b b4 24 94 00 00 	mov    0x94(%rsp),%r14d
  400d9c:	00 
  400d9d:	c7 44 24 70 64 00 00 	movl   $0x64,0x70(%rsp)
  400da4:	00 
  400da5:	89 44 24 10          	mov    %eax,0x10(%rsp)
  400da9:	8b 84 24 8c 00 00 00 	mov    0x8c(%rsp),%eax
  400db0:	44 89 f5             	mov    %r14d,%ebp
  400db3:	89 44 24 64          	mov    %eax,0x64(%rsp)
  400db7:	41 0f af c6          	imul   %r14d,%eax
  400dbb:	89 44 24 74          	mov    %eax,0x74(%rsp)
  400dbf:	83 e8 01             	sub    $0x1,%eax
  400dc2:	48 8d 04 85 04 00 00 	lea    0x4(,%rax,4),%rax
  400dc9:	00 
  400dca:	48 89 44 24 78       	mov    %rax,0x78(%rsp)
  400dcf:	49 63 c6             	movslq %r14d,%rax
  400dd2:	48 89 44 24 20       	mov    %rax,0x20(%rsp)
  400dd7:	41 8d 46 ff          	lea    -0x1(%r14),%eax
  400ddb:	4d 89 e6             	mov    %r12,%r14
  400dde:	89 44 24 50          	mov    %eax,0x50(%rsp)
  400de2:	8b 4c 24 74          	mov    0x74(%rsp),%ecx
  400de6:	85 c9                	test   %ecx,%ecx
  400de8:	7e 0f                	jle    400df9 <main+0x229>
  400dea:	48 8b 54 24 78       	mov    0x78(%rsp),%rdx
  400def:	31 f6                	xor    %esi,%esi
  400df1:	4c 89 ff             	mov    %r15,%rdi
  400df4:	e8 a7 fc ff ff       	callq  400aa0 <memset@plt>
  400df9:	8b 54 24 64          	mov    0x64(%rsp),%edx
  400dfd:	85 d2                	test   %edx,%edx
  400dff:	0f 8e 81 02 00 00    	jle    401086 <main+0x4b6>
  400e05:	8b 44 24 10          	mov    0x10(%rsp),%eax
  400e09:	85 c0                	test   %eax,%eax
  400e0b:	0f 8e 75 02 00 00    	jle    401086 <main+0x4b6>
  400e11:	45 31 ed             	xor    %r13d,%r13d
  400e14:	c7 44 24 60 00 00 00 	movl   $0x0,0x60(%rsp)
  400e1b:	00 
  400e1c:	c7 44 24 54 00 00 00 	movl   $0x0,0x54(%rsp)
  400e23:	00 
  400e24:	45 89 ec             	mov    %r13d,%r12d
  400e27:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  400e2e:	00 00 
  400e30:	85 ed                	test   %ebp,%ebp
  400e32:	0f 8e 30 02 00 00    	jle    401068 <main+0x498>
  400e38:	48 63 44 24 60       	movslq 0x60(%rsp),%rax
  400e3d:	48 8b 5c 24 58       	mov    0x58(%rsp),%rbx
  400e42:	bf 03 00 00 00       	mov    $0x3,%edi
  400e47:	48 8d 04 83          	lea    (%rbx,%rax,4),%rax
  400e4b:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
  400e50:	49 63 c4             	movslq %r12d,%rax
  400e53:	48 89 44 24 48       	mov    %rax,0x48(%rsp)
  400e58:	49 8d 04 87          	lea    (%r15,%rax,4),%rax
  400e5c:	48 89 44 24 28       	mov    %rax,0x28(%rsp)
  400e61:	48 c1 e8 02          	shr    $0x2,%rax
  400e65:	48 f7 d8             	neg    %rax
  400e68:	83 e0 03             	and    $0x3,%eax
  400e6b:	39 e8                	cmp    %ebp,%eax
  400e6d:	0f 47 c5             	cmova  %ebp,%eax
  400e70:	45 31 d2             	xor    %r10d,%r10d
  400e73:	45 31 c0             	xor    %r8d,%r8d
  400e76:	89 44 24 14          	mov    %eax,0x14(%rsp)
  400e7a:	41 8d 44 24 01       	lea    0x1(%r12),%eax
  400e7f:	48 98                	cltq   
  400e81:	49 8d 04 87          	lea    (%r15,%rax,4),%rax
  400e85:	48 89 44 24 38       	mov    %rax,0x38(%rsp)
  400e8a:	41 8d 44 24 02       	lea    0x2(%r12),%eax
  400e8f:	48 98                	cltq   
  400e91:	49 8d 04 87          	lea    (%r15,%rax,4),%rax
  400e95:	48 89 44 24 40       	mov    %rax,0x40(%rsp)
  400e9a:	41 8d 44 24 03       	lea    0x3(%r12),%eax
  400e9f:	48 98                	cltq   
  400ea1:	49 8d 04 87          	lea    (%r15,%rax,4),%rax
  400ea5:	48 89 44 24 30       	mov    %rax,0x30(%rsp)
  400eaa:	e9 8a 01 00 00       	jmpq   401039 <main+0x469>
  400eaf:	90                   	nop
  400eb0:	f3 43 0f 10 0c 96    	movss  (%r14,%r10,4),%xmm1
  400eb6:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
  400ebb:	f3 0f 59 c8          	mulss  %xmm0,%xmm1
  400ebf:	83 fe 01             	cmp    $0x1,%esi
  400ec2:	f3 0f 58 08          	addss  (%rax),%xmm1
  400ec6:	f3 0f 11 08          	movss  %xmm1,(%rax)
  400eca:	b8 01 00 00 00       	mov    $0x1,%eax
  400ecf:	74 6b                	je     400f3c <main+0x36c>
  400ed1:	8d 47 fe             	lea    -0x2(%rdi),%eax
  400ed4:	83 fe 02             	cmp    $0x2,%esi
  400ed7:	48 98                	cltq   
  400ed9:	f3 41 0f 10 0c 86    	movss  (%r14,%rax,4),%xmm1
  400edf:	48 8b 44 24 38       	mov    0x38(%rsp),%rax
  400ee4:	f3 0f 59 c8          	mulss  %xmm0,%xmm1
  400ee8:	f3 0f 58 08          	addss  (%rax),%xmm1
  400eec:	f3 0f 11 08          	movss  %xmm1,(%rax)
  400ef0:	b8 02 00 00 00       	mov    $0x2,%eax
  400ef5:	74 45                	je     400f3c <main+0x36c>
  400ef7:	8d 47 ff             	lea    -0x1(%rdi),%eax
  400efa:	83 fe 03             	cmp    $0x3,%esi
  400efd:	48 98                	cltq   
  400eff:	f3 41 0f 10 0c 86    	movss  (%r14,%rax,4),%xmm1
  400f05:	48 8b 44 24 40       	mov    0x40(%rsp),%rax
  400f0a:	f3 0f 59 c8          	mulss  %xmm0,%xmm1
  400f0e:	f3 0f 58 08          	addss  (%rax),%xmm1
  400f12:	f3 0f 11 08          	movss  %xmm1,(%rax)
  400f16:	b8 03 00 00 00       	mov    $0x3,%eax
  400f1b:	74 1f                	je     400f3c <main+0x36c>
  400f1d:	48 63 c7             	movslq %edi,%rax
  400f20:	f3 41 0f 10 0c 86    	movss  (%r14,%rax,4),%xmm1
  400f26:	f3 0f 59 c8          	mulss  %xmm0,%xmm1
  400f2a:	48 8b 44 24 30       	mov    0x30(%rsp),%rax
  400f2f:	f3 0f 58 08          	addss  (%rax),%xmm1
  400f33:	f3 0f 11 08          	movss  %xmm1,(%rax)
  400f37:	b8 04 00 00 00       	mov    $0x4,%eax
  400f3c:	39 f5                	cmp    %esi,%ebp
  400f3e:	0f 84 e3 00 00 00    	je     401027 <main+0x457>
  400f44:	89 eb                	mov    %ebp,%ebx
  400f46:	89 f1                	mov    %esi,%ecx
  400f48:	29 f3                	sub    %esi,%ebx
  400f4a:	8d 53 fc             	lea    -0x4(%rbx),%edx
  400f4d:	c1 ea 02             	shr    $0x2,%edx
  400f50:	83 c2 01             	add    $0x1,%edx
  400f53:	44 8d 1c 95 00 00 00 	lea    0x0(,%rdx,4),%r11d
  400f5a:	00 
  400f5b:	44 89 5c 24 0c       	mov    %r11d,0xc(%rsp)
  400f60:	44 8b 5c 24 50       	mov    0x50(%rsp),%r11d
  400f65:	41 29 f3             	sub    %esi,%r11d
  400f68:	41 83 fb 02          	cmp    $0x2,%r11d
  400f6c:	76 46                	jbe    400fb4 <main+0x3e4>
  400f6e:	48 8b 74 24 48       	mov    0x48(%rsp),%rsi
  400f73:	0f 28 d0             	movaps %xmm0,%xmm2
  400f76:	0f c6 d2 00          	shufps $0x0,%xmm2,%xmm2
  400f7a:	48 01 ce             	add    %rcx,%rsi
  400f7d:	4c 01 d1             	add    %r10,%rcx
  400f80:	4d 8d 1c b7          	lea    (%r15,%rsi,4),%r11
  400f84:	4d 8d 2c 8e          	lea    (%r14,%rcx,4),%r13
  400f88:	31 f6                	xor    %esi,%esi
  400f8a:	31 c9                	xor    %ecx,%ecx
  400f8c:	41 0f 10 4c 0d 00    	movups 0x0(%r13,%rcx,1),%xmm1
  400f92:	83 c6 01             	add    $0x1,%esi
  400f95:	0f 59 ca             	mulps  %xmm2,%xmm1
  400f98:	41 0f 58 0c 0b       	addps  (%r11,%rcx,1),%xmm1
  400f9d:	41 0f 29 0c 0b       	movaps %xmm1,(%r11,%rcx,1)
  400fa2:	48 83 c1 10          	add    $0x10,%rcx
  400fa6:	39 f2                	cmp    %esi,%edx
  400fa8:	77 e2                	ja     400f8c <main+0x3bc>
  400faa:	8b 4c 24 0c          	mov    0xc(%rsp),%ecx
  400fae:	01 c8                	add    %ecx,%eax
  400fb0:	39 d9                	cmp    %ebx,%ecx
  400fb2:	74 73                	je     401027 <main+0x457>
  400fb4:	42 8d 14 20          	lea    (%rax,%r12,1),%edx
  400fb8:	48 63 d2             	movslq %edx,%rdx
  400fbb:	49 8d 0c 97          	lea    (%r15,%rdx,4),%rcx
  400fbf:	42 8d 14 08          	lea    (%rax,%r9,1),%edx
  400fc3:	48 63 d2             	movslq %edx,%rdx
  400fc6:	f3 41 0f 10 0c 96    	movss  (%r14,%rdx,4),%xmm1
  400fcc:	8d 50 01             	lea    0x1(%rax),%edx
  400fcf:	f3 0f 59 c8          	mulss  %xmm0,%xmm1
  400fd3:	39 ea                	cmp    %ebp,%edx
  400fd5:	f3 0f 58 09          	addss  (%rcx),%xmm1
  400fd9:	f3 0f 11 09          	movss  %xmm1,(%rcx)
  400fdd:	7d 48                	jge    401027 <main+0x457>
  400fdf:	42 8d 0c 22          	lea    (%rdx,%r12,1),%ecx
  400fe3:	44 01 ca             	add    %r9d,%edx
  400fe6:	83 c0 02             	add    $0x2,%eax
  400fe9:	48 63 d2             	movslq %edx,%rdx
  400fec:	39 c5                	cmp    %eax,%ebp
  400fee:	f3 41 0f 10 0c 96    	movss  (%r14,%rdx,4),%xmm1
  400ff4:	48 63 c9             	movslq %ecx,%rcx
  400ff7:	f3 0f 59 c8          	mulss  %xmm0,%xmm1
  400ffb:	49 8d 0c 8f          	lea    (%r15,%rcx,4),%rcx
  400fff:	f3 0f 58 09          	addss  (%rcx),%xmm1
  401003:	f3 0f 11 09          	movss  %xmm1,(%rcx)
  401007:	7e 1e                	jle    401027 <main+0x457>
  401009:	41 8d 14 04          	lea    (%r12,%rax,1),%edx
  40100d:	44 01 c8             	add    %r9d,%eax
  401010:	48 98                	cltq   
  401012:	f3 41 0f 59 04 86    	mulss  (%r14,%rax,4),%xmm0
  401018:	48 63 d2             	movslq %edx,%rdx
  40101b:	49 8d 14 97          	lea    (%r15,%rdx,4),%rdx
  40101f:	f3 0f 58 02          	addss  (%rdx),%xmm0
  401023:	f3 0f 11 02          	movss  %xmm0,(%rdx)
  401027:	49 83 c0 01          	add    $0x1,%r8
  40102b:	01 ef                	add    %ebp,%edi
  40102d:	4c 03 54 24 20       	add    0x20(%rsp),%r10
  401032:	44 39 44 24 10       	cmp    %r8d,0x10(%rsp)
  401037:	7e 2f                	jle    401068 <main+0x498>
  401039:	8b 74 24 14          	mov    0x14(%rsp),%esi
  40103d:	83 fd 04             	cmp    $0x4,%ebp
  401040:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  401045:	44 8d 4f fd          	lea    -0x3(%rdi),%r9d
  401049:	0f 46 f5             	cmovbe %ebp,%esi
  40104c:	f3 42 0f 10 04 80    	movss  (%rax,%r8,4),%xmm0
  401052:	85 f6                	test   %esi,%esi
  401054:	0f 85 56 fe ff ff    	jne    400eb0 <main+0x2e0>
  40105a:	31 c0                	xor    %eax,%eax
  40105c:	e9 e3 fe ff ff       	jmpq   400f44 <main+0x374>
  401061:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  401068:	83 44 24 54 01       	addl   $0x1,0x54(%rsp)
  40106d:	41 01 ec             	add    %ebp,%r12d
  401070:	8b 7c 24 10          	mov    0x10(%rsp),%edi
  401074:	8b 44 24 54          	mov    0x54(%rsp),%eax
  401078:	01 7c 24 60          	add    %edi,0x60(%rsp)
  40107c:	3b 44 24 64          	cmp    0x64(%rsp),%eax
  401080:	0f 85 aa fd ff ff    	jne    400e30 <main+0x260>
  401086:	83 6c 24 70 01       	subl   $0x1,0x70(%rsp)
  40108b:	0f 85 51 fd ff ff    	jne    400de2 <main+0x212>
  401091:	48 8d bc 24 b0 00 00 	lea    0xb0(%rsp),%rdi
  401098:	00 
  401099:	31 f6                	xor    %esi,%esi
  40109b:	e8 e0 fa ff ff       	callq  400b80 <gettimeofday@plt>
  4010a0:	66 0f ef c0          	pxor   %xmm0,%xmm0
  4010a4:	bf 46 2e 40 00       	mov    $0x402e46,%edi
  4010a9:	f2 0f 10 15 4f 1e 00 	movsd  0x1e4f(%rip),%xmm2        # 402f00 <_IO_stdin_used+0x140>
  4010b0:	00 
  4010b1:	b8 01 00 00 00       	mov    $0x1,%eax
  4010b6:	66 0f ef c9          	pxor   %xmm1,%xmm1
  4010ba:	f2 48 0f 2a 84 24 b8 	cvtsi2sdq 0xb8(%rsp),%xmm0
  4010c1:	00 00 00 
  4010c4:	f2 0f 5e c2          	divsd  %xmm2,%xmm0
  4010c8:	f2 48 0f 2a 8c 24 b0 	cvtsi2sdq 0xb0(%rsp),%xmm1
  4010cf:	00 00 00 
  4010d2:	f2 0f 58 c1          	addsd  %xmm1,%xmm0
  4010d6:	66 0f ef c9          	pxor   %xmm1,%xmm1
  4010da:	f2 48 0f 2a 8c 24 a8 	cvtsi2sdq 0xa8(%rsp),%xmm1
  4010e1:	00 00 00 
  4010e4:	f2 0f 5e ca          	divsd  %xmm2,%xmm1
  4010e8:	66 0f ef d2          	pxor   %xmm2,%xmm2
  4010ec:	f2 48 0f 2a 94 24 a0 	cvtsi2sdq 0xa0(%rsp),%xmm2
  4010f3:	00 00 00 
  4010f6:	f2 0f 58 ca          	addsd  %xmm2,%xmm1
  4010fa:	f2 0f 5c c1          	subsd  %xmm1,%xmm0
  4010fe:	f2 0f 5e 05 02 1e 00 	divsd  0x1e02(%rip),%xmm0        # 402f08 <_IO_stdin_used+0x148>
  401105:	00 
  401106:	e8 65 f9 ff ff       	callq  400a70 <printf@plt>
  40110b:	48 8b 44 24 68       	mov    0x68(%rsp),%rax
  401110:	be 4c 2e 40 00       	mov    $0x402e4c,%esi
  401115:	48 8b 78 18          	mov    0x18(%rax),%rdi
  401119:	e8 d2 f9 ff ff       	callq  400af0 <fopen@plt>
  40111e:	48 85 c0             	test   %rax,%rax
  401121:	48 89 c3             	mov    %rax,%rbx
  401124:	0f 84 17 01 00 00    	je     401241 <main+0x671>
  40112a:	8b 94 24 94 00 00 00 	mov    0x94(%rsp),%edx
  401131:	8b b4 24 8c 00 00 00 	mov    0x8c(%rsp),%esi
  401138:	4c 89 f9             	mov    %r15,%rcx
  40113b:	48 89 c7             	mov    %rax,%rdi
  40113e:	e8 ad 05 00 00       	callq  4016f0 <_Z12write_sparseP8_IO_FILEiiPKf>
  401143:	48 89 df             	mov    %rbx,%rdi
  401146:	e8 95 f9 ff ff       	callq  400ae0 <fclose@plt>
  40114b:	48 8b 7c 24 58       	mov    0x58(%rsp),%rdi
  401150:	e8 ab f9 ff ff       	callq  400b00 <free@plt>
  401155:	4c 89 f7             	mov    %r14,%rdi
  401158:	e8 a3 f9 ff ff       	callq  400b00 <free@plt>
  40115d:	4c 89 ff             	mov    %r15,%rdi
  401160:	e8 9b f9 ff ff       	callq  400b00 <free@plt>
  401165:	48 81 c4 c8 00 00 00 	add    $0xc8,%rsp
  40116c:	31 c0                	xor    %eax,%eax
  40116e:	5b                   	pop    %rbx
  40116f:	5d                   	pop    %rbp
  401170:	41 5c                	pop    %r12
  401172:	41 5d                	pop    %r13
  401174:	41 5e                	pop    %r14
  401176:	41 5f                	pop    %r15
  401178:	c3                   	retq   
  401179:	bf 33 2e 40 00       	mov    $0x402e33,%edi
  40117e:	e8 dd f9 ff ff       	callq  400b60 <puts@plt>
  401183:	bf 01 00 00 00       	mov    $0x1,%edi
  401188:	e8 83 f9 ff ff       	callq  400b10 <exit@plt>
  40118d:	8b 94 24 94 00 00 00 	mov    0x94(%rsp),%edx
  401194:	8b b4 24 90 00 00 00 	mov    0x90(%rsp),%esi
  40119b:	4c 89 e1             	mov    %r12,%rcx
  40119e:	48 89 df             	mov    %rbx,%rdi
  4011a1:	e8 ea 08 00 00       	callq  401a90 <_Z10read_denseP8_IO_FILEiiPf>
  4011a6:	e9 77 fb ff ff       	jmpq   400d22 <main+0x152>
  4011ab:	bf f3 2d 40 00       	mov    $0x402df3,%edi
  4011b0:	e8 ab f9 ff ff       	callq  400b60 <puts@plt>
  4011b5:	48 89 ef             	mov    %rbp,%rdi
  4011b8:	e8 23 f9 ff ff       	callq  400ae0 <fclose@plt>
  4011bd:	48 89 df             	mov    %rbx,%rdi
  4011c0:	e8 1b f9 ff ff       	callq  400ae0 <fclose@plt>
  4011c5:	bf 01 00 00 00       	mov    $0x1,%edi
  4011ca:	e8 41 f9 ff ff       	callq  400b10 <exit@plt>
  4011cf:	bf 02 00 00 00       	mov    $0x2,%edi
  4011d4:	e8 37 f9 ff ff       	callq  400b10 <exit@plt>
  4011d9:	48 8b 16             	mov    (%rsi),%rdx
  4011dc:	48 8b 3d 0d 2f 20 00 	mov    0x202f0d(%rip),%rdi        # 6040f0 <stderr@@GLIBC_2.2.5>
  4011e3:	be b0 2e 40 00       	mov    $0x402eb0,%esi
  4011e8:	31 c0                	xor    %eax,%eax
  4011ea:	e8 51 f9 ff ff       	callq  400b40 <fprintf@plt>
  4011ef:	bf 01 00 00 00       	mov    $0x1,%edi
  4011f4:	e8 17 f9 ff ff       	callq  400b10 <exit@plt>
  4011f9:	bf 21 2e 40 00       	mov    $0x402e21,%edi
  4011fe:	e8 5d f9 ff ff       	callq  400b60 <puts@plt>
  401203:	bf 01 00 00 00       	mov    $0x1,%edi
  401208:	e8 03 f9 ff ff       	callq  400b10 <exit@plt>
  40120d:	bf 0f 2e 40 00       	mov    $0x402e0f,%edi
  401212:	e8 49 f9 ff ff       	callq  400b60 <puts@plt>
  401217:	bf 01 00 00 00       	mov    $0x1,%edi
  40121c:	e8 ef f8 ff ff       	callq  400b10 <exit@plt>
  401221:	48 8b 4c 24 58       	mov    0x58(%rsp),%rcx
  401226:	8b 94 24 90 00 00 00 	mov    0x90(%rsp),%edx
  40122d:	48 89 ef             	mov    %rbp,%rdi
  401230:	8b b4 24 8c 00 00 00 	mov    0x8c(%rsp),%esi
  401237:	e8 54 08 00 00       	callq  401a90 <_Z10read_denseP8_IO_FILEiiPf>
  40123c:	e9 b9 fa ff ff       	jmpq   400cfa <main+0x12a>
  401241:	bf 03 00 00 00       	mov    $0x3,%edi
  401246:	e8 c5 f8 ff ff       	callq  400b10 <exit@plt>

000000000040124b <_start>:
  40124b:	31 ed                	xor    %ebp,%ebp
  40124d:	49 89 d1             	mov    %rdx,%r9
  401250:	5e                   	pop    %rsi
  401251:	48 89 e2             	mov    %rsp,%rdx
  401254:	48 83 e4 f0          	and    $0xfffffffffffffff0,%rsp
  401258:	50                   	push   %rax
  401259:	54                   	push   %rsp
  40125a:	49 c7 c0 b0 2d 40 00 	mov    $0x402db0,%r8
  401261:	48 c7 c1 40 2d 40 00 	mov    $0x402d40,%rcx
  401268:	48 c7 c7 d0 0b 40 00 	mov    $0x400bd0,%rdi
  40126f:	e8 1c f9 ff ff       	callq  400b90 <__libc_start_main@plt>
  401274:	f4                   	hlt    
  401275:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40127c:	00 00 00 
  40127f:	90                   	nop

0000000000401280 <deregister_tm_clones>:
  401280:	b8 df 40 60 00       	mov    $0x6040df,%eax
  401285:	55                   	push   %rbp
  401286:	48 2d d8 40 60 00    	sub    $0x6040d8,%rax
  40128c:	48 83 f8 0e          	cmp    $0xe,%rax
  401290:	48 89 e5             	mov    %rsp,%rbp
  401293:	76 1b                	jbe    4012b0 <deregister_tm_clones+0x30>
  401295:	b8 00 00 00 00       	mov    $0x0,%eax
  40129a:	48 85 c0             	test   %rax,%rax
  40129d:	74 11                	je     4012b0 <deregister_tm_clones+0x30>
  40129f:	5d                   	pop    %rbp
  4012a0:	bf d8 40 60 00       	mov    $0x6040d8,%edi
  4012a5:	ff e0                	jmpq   *%rax
  4012a7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  4012ae:	00 00 
  4012b0:	5d                   	pop    %rbp
  4012b1:	c3                   	retq   
  4012b2:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  4012b9:	1f 84 00 00 00 00 00 

00000000004012c0 <register_tm_clones>:
  4012c0:	be d8 40 60 00       	mov    $0x6040d8,%esi
  4012c5:	55                   	push   %rbp
  4012c6:	48 81 ee d8 40 60 00 	sub    $0x6040d8,%rsi
  4012cd:	48 c1 fe 03          	sar    $0x3,%rsi
  4012d1:	48 89 e5             	mov    %rsp,%rbp
  4012d4:	48 89 f0             	mov    %rsi,%rax
  4012d7:	48 c1 e8 3f          	shr    $0x3f,%rax
  4012db:	48 01 c6             	add    %rax,%rsi
  4012de:	48 d1 fe             	sar    %rsi
  4012e1:	74 15                	je     4012f8 <register_tm_clones+0x38>
  4012e3:	b8 00 00 00 00       	mov    $0x0,%eax
  4012e8:	48 85 c0             	test   %rax,%rax
  4012eb:	74 0b                	je     4012f8 <register_tm_clones+0x38>
  4012ed:	5d                   	pop    %rbp
  4012ee:	bf d8 40 60 00       	mov    $0x6040d8,%edi
  4012f3:	ff e0                	jmpq   *%rax
  4012f5:	0f 1f 00             	nopl   (%rax)
  4012f8:	5d                   	pop    %rbp
  4012f9:	c3                   	retq   
  4012fa:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000401300 <__do_global_dtors_aux>:
  401300:	80 3d f1 2d 20 00 00 	cmpb   $0x0,0x202df1(%rip)        # 6040f8 <completed.6893>
  401307:	75 11                	jne    40131a <__do_global_dtors_aux+0x1a>
  401309:	55                   	push   %rbp
  40130a:	48 89 e5             	mov    %rsp,%rbp
  40130d:	e8 6e ff ff ff       	callq  401280 <deregister_tm_clones>
  401312:	5d                   	pop    %rbp
  401313:	c6 05 de 2d 20 00 01 	movb   $0x1,0x202dde(%rip)        # 6040f8 <completed.6893>
  40131a:	f3 c3                	repz retq 
  40131c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000401320 <frame_dummy>:
  401320:	bf d0 3d 60 00       	mov    $0x603dd0,%edi
  401325:	48 83 3f 00          	cmpq   $0x0,(%rdi)
  401329:	75 05                	jne    401330 <frame_dummy+0x10>
  40132b:	eb 93                	jmp    4012c0 <register_tm_clones>
  40132d:	0f 1f 00             	nopl   (%rax)
  401330:	b8 00 00 00 00       	mov    $0x0,%eax
  401335:	48 85 c0             	test   %rax,%rax
  401338:	74 f1                	je     40132b <frame_dummy+0xb>
  40133a:	55                   	push   %rbp
  40133b:	48 89 e5             	mov    %rsp,%rbp
  40133e:	ff d0                	callq  *%rax
  401340:	5d                   	pop    %rbp
  401341:	e9 7a ff ff ff       	jmpq   4012c0 <register_tm_clones>
  401346:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40134d:	00 00 00 

0000000000401350 <_Z3cmpPfS_i>:
  401350:	85 d2                	test   %edx,%edx
  401352:	7e 32                	jle    401386 <_Z3cmpPfS_i+0x36>
  401354:	f3 0f 10 07          	movss  (%rdi),%xmm0
  401358:	0f 2e 06             	ucomiss (%rsi),%xmm0
  40135b:	7a 33                	jp     401390 <_Z3cmpPfS_i+0x40>
  40135d:	75 31                	jne    401390 <_Z3cmpPfS_i+0x40>
  40135f:	b8 01 00 00 00       	mov    $0x1,%eax
  401364:	eb 1c                	jmp    401382 <_Z3cmpPfS_i+0x32>
  401366:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40136d:	00 00 00 
  401370:	f3 0f 10 04 87       	movss  (%rdi,%rax,4),%xmm0
  401375:	48 83 c0 01          	add    $0x1,%rax
  401379:	0f 2e 44 86 fc       	ucomiss -0x4(%rsi,%rax,4),%xmm0
  40137e:	7a 10                	jp     401390 <_Z3cmpPfS_i+0x40>
  401380:	75 0e                	jne    401390 <_Z3cmpPfS_i+0x40>
  401382:	39 c2                	cmp    %eax,%edx
  401384:	7f ea                	jg     401370 <_Z3cmpPfS_i+0x20>
  401386:	bf d6 2d 40 00       	mov    $0x402dd6,%edi
  40138b:	e9 d0 f7 ff ff       	jmpq   400b60 <puts@plt>
  401390:	bf c4 2d 40 00       	mov    $0x402dc4,%edi
  401395:	e9 c6 f7 ff ff       	jmpq   400b60 <puts@plt>
  40139a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

00000000004013a0 <_Z12generate_matiiiPfS_>:
  4013a0:	0f af fe             	imul   %esi,%edi
  4013a3:	41 54                	push   %r12
  4013a5:	55                   	push   %rbp
  4013a6:	53                   	push   %rbx
  4013a7:	85 ff                	test   %edi,%edi
  4013a9:	0f 8e f7 00 00 00    	jle    4014a6 <_Z12generate_matiiiPfS_+0x106>
  4013af:	48 89 c8             	mov    %rcx,%rax
  4013b2:	48 c1 e8 02          	shr    $0x2,%rax
  4013b6:	48 f7 d8             	neg    %rax
  4013b9:	83 e0 03             	and    $0x3,%eax
  4013bc:	39 f8                	cmp    %edi,%eax
  4013be:	0f 47 c7             	cmova  %edi,%eax
  4013c1:	83 ff 06             	cmp    $0x6,%edi
  4013c4:	0f 8f f6 01 00 00    	jg     4015c0 <_Z12generate_matiiiPfS_+0x220>
  4013ca:	89 f8                	mov    %edi,%eax
  4013cc:	f3 0f 10 05 0c 1b 00 	movss  0x1b0c(%rip),%xmm0        # 402ee0 <_IO_stdin_used+0x120>
  4013d3:	00 
  4013d4:	83 f8 01             	cmp    $0x1,%eax
  4013d7:	f3 0f 11 01          	movss  %xmm0,(%rcx)
  4013db:	0f 84 5f 02 00 00    	je     401640 <_Z12generate_matiiiPfS_+0x2a0>
  4013e1:	83 f8 02             	cmp    $0x2,%eax
  4013e4:	f3 0f 11 41 04       	movss  %xmm0,0x4(%rcx)
  4013e9:	0f 84 31 02 00 00    	je     401620 <_Z12generate_matiiiPfS_+0x280>
  4013ef:	83 f8 03             	cmp    $0x3,%eax
  4013f2:	f3 0f 11 41 08       	movss  %xmm0,0x8(%rcx)
  4013f7:	0f 84 33 02 00 00    	je     401630 <_Z12generate_matiiiPfS_+0x290>
  4013fd:	83 f8 04             	cmp    $0x4,%eax
  401400:	f3 0f 11 41 0c       	movss  %xmm0,0xc(%rcx)
  401405:	0f 84 55 02 00 00    	je     401660 <_Z12generate_matiiiPfS_+0x2c0>
  40140b:	83 f8 05             	cmp    $0x5,%eax
  40140e:	f3 0f 11 41 10       	movss  %xmm0,0x10(%rcx)
  401413:	0f 84 b7 01 00 00    	je     4015d0 <_Z12generate_matiiiPfS_+0x230>
  401419:	f3 0f 11 41 14       	movss  %xmm0,0x14(%rcx)
  40141e:	41 b9 06 00 00 00    	mov    $0x6,%r9d
  401424:	39 c7                	cmp    %eax,%edi
  401426:	74 7e                	je     4014a6 <_Z12generate_matiiiPfS_+0x106>
  401428:	89 fb                	mov    %edi,%ebx
  40142a:	44 8d 5f ff          	lea    -0x1(%rdi),%r11d
  40142e:	41 89 c4             	mov    %eax,%r12d
  401431:	29 c3                	sub    %eax,%ebx
  401433:	44 8d 53 fc          	lea    -0x4(%rbx),%r10d
  401437:	41 29 c3             	sub    %eax,%r11d
  40143a:	41 c1 ea 02          	shr    $0x2,%r10d
  40143e:	41 83 c2 01          	add    $0x1,%r10d
  401442:	41 83 fb 02          	cmp    $0x2,%r11d
  401446:	42 8d 2c 95 00 00 00 	lea    0x0(,%r10,4),%ebp
  40144d:	00 
  40144e:	76 25                	jbe    401475 <_Z12generate_matiiiPfS_+0xd5>
  401450:	4e 8d 1c a1          	lea    (%rcx,%r12,4),%r11
  401454:	0f 28 05 95 1a 00 00 	movaps 0x1a95(%rip),%xmm0        # 402ef0 <_IO_stdin_used+0x130>
  40145b:	31 c0                	xor    %eax,%eax
  40145d:	83 c0 01             	add    $0x1,%eax
  401460:	49 83 c3 10          	add    $0x10,%r11
  401464:	41 0f 29 43 f0       	movaps %xmm0,-0x10(%r11)
  401469:	41 39 c2             	cmp    %eax,%r10d
  40146c:	77 ef                	ja     40145d <_Z12generate_matiiiPfS_+0xbd>
  40146e:	41 01 e9             	add    %ebp,%r9d
  401471:	39 eb                	cmp    %ebp,%ebx
  401473:	74 31                	je     4014a6 <_Z12generate_matiiiPfS_+0x106>
  401475:	49 63 c1             	movslq %r9d,%rax
  401478:	f3 0f 10 05 60 1a 00 	movss  0x1a60(%rip),%xmm0        # 402ee0 <_IO_stdin_used+0x120>
  40147f:	00 
  401480:	f3 0f 11 04 81       	movss  %xmm0,(%rcx,%rax,4)
  401485:	41 8d 41 01          	lea    0x1(%r9),%eax
  401489:	39 c7                	cmp    %eax,%edi
  40148b:	7e 19                	jle    4014a6 <_Z12generate_matiiiPfS_+0x106>
  40148d:	41 83 c1 02          	add    $0x2,%r9d
  401491:	48 98                	cltq   
  401493:	44 39 cf             	cmp    %r9d,%edi
  401496:	f3 0f 11 04 81       	movss  %xmm0,(%rcx,%rax,4)
  40149b:	7e 09                	jle    4014a6 <_Z12generate_matiiiPfS_+0x106>
  40149d:	4d 63 c9             	movslq %r9d,%r9
  4014a0:	f3 42 0f 11 04 89    	movss  %xmm0,(%rcx,%r9,4)
  4014a6:	0f af d6             	imul   %esi,%edx
  4014a9:	85 d2                	test   %edx,%edx
  4014ab:	0f 8e f7 00 00 00    	jle    4015a8 <_Z12generate_matiiiPfS_+0x208>
  4014b1:	4c 89 c0             	mov    %r8,%rax
  4014b4:	48 c1 e8 02          	shr    $0x2,%rax
  4014b8:	48 f7 d8             	neg    %rax
  4014bb:	83 e0 03             	and    $0x3,%eax
  4014be:	39 d0                	cmp    %edx,%eax
  4014c0:	0f 47 c2             	cmova  %edx,%eax
  4014c3:	83 fa 06             	cmp    $0x6,%edx
  4014c6:	0f 8f e4 00 00 00    	jg     4015b0 <_Z12generate_matiiiPfS_+0x210>
  4014cc:	89 d0                	mov    %edx,%eax
  4014ce:	f3 0f 10 05 0a 1a 00 	movss  0x1a0a(%rip),%xmm0        # 402ee0 <_IO_stdin_used+0x120>
  4014d5:	00 
  4014d6:	83 f8 01             	cmp    $0x1,%eax
  4014d9:	f3 41 0f 11 00       	movss  %xmm0,(%r8)
  4014de:	0f 84 fc 00 00 00    	je     4015e0 <_Z12generate_matiiiPfS_+0x240>
  4014e4:	83 f8 02             	cmp    $0x2,%eax
  4014e7:	f3 41 0f 11 40 04    	movss  %xmm0,0x4(%r8)
  4014ed:	0f 84 fd 00 00 00    	je     4015f0 <_Z12generate_matiiiPfS_+0x250>
  4014f3:	83 f8 03             	cmp    $0x3,%eax
  4014f6:	f3 41 0f 11 40 08    	movss  %xmm0,0x8(%r8)
  4014fc:	0f 84 fe 00 00 00    	je     401600 <_Z12generate_matiiiPfS_+0x260>
  401502:	83 f8 04             	cmp    $0x4,%eax
  401505:	f3 41 0f 11 40 0c    	movss  %xmm0,0xc(%r8)
  40150b:	0f 84 ff 00 00 00    	je     401610 <_Z12generate_matiiiPfS_+0x270>
  401511:	83 f8 05             	cmp    $0x5,%eax
  401514:	f3 41 0f 11 40 10    	movss  %xmm0,0x10(%r8)
  40151a:	0f 84 30 01 00 00    	je     401650 <_Z12generate_matiiiPfS_+0x2b0>
  401520:	f3 41 0f 11 40 14    	movss  %xmm0,0x14(%r8)
  401526:	b9 06 00 00 00       	mov    $0x6,%ecx
  40152b:	39 c2                	cmp    %eax,%edx
  40152d:	74 79                	je     4015a8 <_Z12generate_matiiiPfS_+0x208>
  40152f:	41 89 d1             	mov    %edx,%r9d
  401532:	8d 7a ff             	lea    -0x1(%rdx),%edi
  401535:	41 89 c3             	mov    %eax,%r11d
  401538:	41 29 c1             	sub    %eax,%r9d
  40153b:	41 8d 71 fc          	lea    -0x4(%r9),%esi
  40153f:	29 c7                	sub    %eax,%edi
  401541:	c1 ee 02             	shr    $0x2,%esi
  401544:	83 c6 01             	add    $0x1,%esi
  401547:	83 ff 02             	cmp    $0x2,%edi
  40154a:	44 8d 14 b5 00 00 00 	lea    0x0(,%rsi,4),%r10d
  401551:	00 
  401552:	76 24                	jbe    401578 <_Z12generate_matiiiPfS_+0x1d8>
  401554:	4b 8d 3c 98          	lea    (%r8,%r11,4),%rdi
  401558:	0f 28 05 91 19 00 00 	movaps 0x1991(%rip),%xmm0        # 402ef0 <_IO_stdin_used+0x130>
  40155f:	31 c0                	xor    %eax,%eax
  401561:	83 c0 01             	add    $0x1,%eax
  401564:	48 83 c7 10          	add    $0x10,%rdi
  401568:	0f 29 47 f0          	movaps %xmm0,-0x10(%rdi)
  40156c:	39 c6                	cmp    %eax,%esi
  40156e:	77 f1                	ja     401561 <_Z12generate_matiiiPfS_+0x1c1>
  401570:	44 01 d1             	add    %r10d,%ecx
  401573:	45 39 d1             	cmp    %r10d,%r9d
  401576:	74 30                	je     4015a8 <_Z12generate_matiiiPfS_+0x208>
  401578:	48 63 c1             	movslq %ecx,%rax
  40157b:	f3 0f 10 05 5d 19 00 	movss  0x195d(%rip),%xmm0        # 402ee0 <_IO_stdin_used+0x120>
  401582:	00 
  401583:	f3 41 0f 11 04 80    	movss  %xmm0,(%r8,%rax,4)
  401589:	8d 41 01             	lea    0x1(%rcx),%eax
  40158c:	39 c2                	cmp    %eax,%edx
  40158e:	7e 18                	jle    4015a8 <_Z12generate_matiiiPfS_+0x208>
  401590:	83 c1 02             	add    $0x2,%ecx
  401593:	48 98                	cltq   
  401595:	39 d1                	cmp    %edx,%ecx
  401597:	f3 41 0f 11 04 80    	movss  %xmm0,(%r8,%rax,4)
  40159d:	7d 09                	jge    4015a8 <_Z12generate_matiiiPfS_+0x208>
  40159f:	48 63 c9             	movslq %ecx,%rcx
  4015a2:	f3 41 0f 11 04 88    	movss  %xmm0,(%r8,%rcx,4)
  4015a8:	5b                   	pop    %rbx
  4015a9:	5d                   	pop    %rbp
  4015aa:	41 5c                	pop    %r12
  4015ac:	c3                   	retq   
  4015ad:	0f 1f 00             	nopl   (%rax)
  4015b0:	85 c0                	test   %eax,%eax
  4015b2:	0f 85 16 ff ff ff    	jne    4014ce <_Z12generate_matiiiPfS_+0x12e>
  4015b8:	31 c9                	xor    %ecx,%ecx
  4015ba:	e9 70 ff ff ff       	jmpq   40152f <_Z12generate_matiiiPfS_+0x18f>
  4015bf:	90                   	nop
  4015c0:	85 c0                	test   %eax,%eax
  4015c2:	0f 85 04 fe ff ff    	jne    4013cc <_Z12generate_matiiiPfS_+0x2c>
  4015c8:	45 31 c9             	xor    %r9d,%r9d
  4015cb:	e9 58 fe ff ff       	jmpq   401428 <_Z12generate_matiiiPfS_+0x88>
  4015d0:	41 b9 05 00 00 00    	mov    $0x5,%r9d
  4015d6:	e9 49 fe ff ff       	jmpq   401424 <_Z12generate_matiiiPfS_+0x84>
  4015db:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  4015e0:	b9 01 00 00 00       	mov    $0x1,%ecx
  4015e5:	e9 41 ff ff ff       	jmpq   40152b <_Z12generate_matiiiPfS_+0x18b>
  4015ea:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  4015f0:	b9 02 00 00 00       	mov    $0x2,%ecx
  4015f5:	e9 31 ff ff ff       	jmpq   40152b <_Z12generate_matiiiPfS_+0x18b>
  4015fa:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  401600:	b9 03 00 00 00       	mov    $0x3,%ecx
  401605:	e9 21 ff ff ff       	jmpq   40152b <_Z12generate_matiiiPfS_+0x18b>
  40160a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  401610:	b9 04 00 00 00       	mov    $0x4,%ecx
  401615:	e9 11 ff ff ff       	jmpq   40152b <_Z12generate_matiiiPfS_+0x18b>
  40161a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  401620:	41 b9 02 00 00 00    	mov    $0x2,%r9d
  401626:	e9 f9 fd ff ff       	jmpq   401424 <_Z12generate_matiiiPfS_+0x84>
  40162b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  401630:	41 b9 03 00 00 00    	mov    $0x3,%r9d
  401636:	e9 e9 fd ff ff       	jmpq   401424 <_Z12generate_matiiiPfS_+0x84>
  40163b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  401640:	41 b9 01 00 00 00    	mov    $0x1,%r9d
  401646:	e9 d9 fd ff ff       	jmpq   401424 <_Z12generate_matiiiPfS_+0x84>
  40164b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  401650:	b9 05 00 00 00       	mov    $0x5,%ecx
  401655:	e9 d1 fe ff ff       	jmpq   40152b <_Z12generate_matiiiPfS_+0x18b>
  40165a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  401660:	41 b9 04 00 00 00    	mov    $0x4,%r9d
  401666:	e9 b9 fd ff ff       	jmpq   401424 <_Z12generate_matiiiPfS_+0x84>
  40166b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000401670 <_Z11read_sparseP8_IO_FILEiiiPf>:
  401670:	85 c9                	test   %ecx,%ecx
  401672:	7e 6d                	jle    4016e1 <_Z11read_sparseP8_IO_FILEiiiPf+0x71>
  401674:	41 56                	push   %r14
  401676:	41 55                	push   %r13
  401678:	45 31 f6             	xor    %r14d,%r14d
  40167b:	41 54                	push   %r12
  40167d:	55                   	push   %rbp
  40167e:	4d 89 c5             	mov    %r8,%r13
  401681:	53                   	push   %rbx
  401682:	41 89 d4             	mov    %edx,%r12d
  401685:	89 cb                	mov    %ecx,%ebx
  401687:	48 89 fd             	mov    %rdi,%rbp
  40168a:	48 83 ec 10          	sub    $0x10,%rsp
  40168e:	66 90                	xchg   %ax,%ax
  401690:	4c 8d 44 24 0c       	lea    0xc(%rsp),%r8
  401695:	48 8d 4c 24 08       	lea    0x8(%rsp),%rcx
  40169a:	48 8d 54 24 04       	lea    0x4(%rsp),%rdx
  40169f:	31 c0                	xor    %eax,%eax
  4016a1:	be e2 2d 40 00       	mov    $0x402de2,%esi
  4016a6:	48 89 ef             	mov    %rbp,%rdi
  4016a9:	e8 02 f5 ff ff       	callq  400bb0 <fscanf@plt>
  4016ae:	8b 44 24 04          	mov    0x4(%rsp),%eax
  4016b2:	41 83 c6 01          	add    $0x1,%r14d
  4016b6:	f3 0f 10 44 24 0c    	movss  0xc(%rsp),%xmm0
  4016bc:	83 e8 01             	sub    $0x1,%eax
  4016bf:	41 0f af c4          	imul   %r12d,%eax
  4016c3:	03 44 24 08          	add    0x8(%rsp),%eax
  4016c7:	44 39 f3             	cmp    %r14d,%ebx
  4016ca:	48 98                	cltq   
  4016cc:	f3 41 0f 11 44 85 fc 	movss  %xmm0,-0x4(%r13,%rax,4)
  4016d3:	75 bb                	jne    401690 <_Z11read_sparseP8_IO_FILEiiiPf+0x20>
  4016d5:	48 83 c4 10          	add    $0x10,%rsp
  4016d9:	5b                   	pop    %rbx
  4016da:	5d                   	pop    %rbp
  4016db:	41 5c                	pop    %r12
  4016dd:	41 5d                	pop    %r13
  4016df:	41 5e                	pop    %r14
  4016e1:	f3 c3                	repz retq 
  4016e3:	0f 1f 00             	nopl   (%rax)
  4016e6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4016ed:	00 00 00 

00000000004016f0 <_Z12write_sparseP8_IO_FILEiiPKf>:
  4016f0:	41 57                	push   %r15
  4016f2:	41 56                	push   %r14
  4016f4:	41 55                	push   %r13
  4016f6:	41 89 f5             	mov    %esi,%r13d
  4016f9:	41 54                	push   %r12
  4016fb:	44 0f af ea          	imul   %edx,%r13d
  4016ff:	55                   	push   %rbp
  401700:	53                   	push   %rbx
  401701:	49 89 fc             	mov    %rdi,%r12
  401704:	89 f3                	mov    %esi,%ebx
  401706:	89 d5                	mov    %edx,%ebp
  401708:	48 83 ec 28          	sub    $0x28,%rsp
  40170c:	45 85 ed             	test   %r13d,%r13d
  40170f:	0f 8e 42 03 00 00    	jle    401a57 <_Z12write_sparseP8_IO_FILEiiPKf+0x367>
  401715:	48 89 c8             	mov    %rcx,%rax
  401718:	49 89 ce             	mov    %rcx,%r14
  40171b:	48 c1 e8 02          	shr    $0x2,%rax
  40171f:	48 f7 d8             	neg    %rax
  401722:	83 e0 03             	and    $0x3,%eax
  401725:	44 39 e8             	cmp    %r13d,%eax
  401728:	41 0f 47 c5          	cmova  %r13d,%eax
  40172c:	41 83 fd 06          	cmp    $0x6,%r13d
  401730:	0f 8f 4a 02 00 00    	jg     401980 <_Z12write_sparseP8_IO_FILEiiPKf+0x290>
  401736:	44 89 e8             	mov    %r13d,%eax
  401739:	66 0f ef c9          	pxor   %xmm1,%xmm1
  40173d:	41 0f 2e 0e          	ucomiss (%r14),%xmm1
  401741:	7a 06                	jp     401749 <_Z12write_sparseP8_IO_FILEiiPKf+0x59>
  401743:	0f 84 17 02 00 00    	je     401960 <_Z12write_sparseP8_IO_FILEiiPKf+0x270>
  401749:	ba 01 00 00 00       	mov    $0x1,%edx
  40174e:	83 f8 01             	cmp    $0x1,%eax
  401751:	41 89 d7             	mov    %edx,%r15d
  401754:	0f 84 14 02 00 00    	je     40196e <_Z12write_sparseP8_IO_FILEiiPKf+0x27e>
  40175a:	41 0f 2e 4e 04       	ucomiss 0x4(%r14),%xmm1
  40175f:	7a 06                	jp     401767 <_Z12write_sparseP8_IO_FILEiiPKf+0x77>
  401761:	0f 84 39 02 00 00    	je     4019a0 <_Z12write_sparseP8_IO_FILEiiPKf+0x2b0>
  401767:	41 bf 01 00 00 00    	mov    $0x1,%r15d
  40176d:	41 01 d7             	add    %edx,%r15d
  401770:	83 f8 02             	cmp    $0x2,%eax
  401773:	0f 84 36 02 00 00    	je     4019af <_Z12write_sparseP8_IO_FILEiiPKf+0x2bf>
  401779:	41 0f 2e 4e 08       	ucomiss 0x8(%r14),%xmm1
  40177e:	7a 06                	jp     401786 <_Z12write_sparseP8_IO_FILEiiPKf+0x96>
  401780:	0f 84 3a 02 00 00    	je     4019c0 <_Z12write_sparseP8_IO_FILEiiPKf+0x2d0>
  401786:	ba 01 00 00 00       	mov    $0x1,%edx
  40178b:	41 01 d7             	add    %edx,%r15d
  40178e:	83 f8 03             	cmp    $0x3,%eax
  401791:	0f 84 37 02 00 00    	je     4019ce <_Z12write_sparseP8_IO_FILEiiPKf+0x2de>
  401797:	41 0f 2e 4e 0c       	ucomiss 0xc(%r14),%xmm1
  40179c:	7a 06                	jp     4017a4 <_Z12write_sparseP8_IO_FILEiiPKf+0xb4>
  40179e:	0f 84 3c 02 00 00    	je     4019e0 <_Z12write_sparseP8_IO_FILEiiPKf+0x2f0>
  4017a4:	ba 01 00 00 00       	mov    $0x1,%edx
  4017a9:	41 01 d7             	add    %edx,%r15d
  4017ac:	83 f8 04             	cmp    $0x4,%eax
  4017af:	0f 84 39 02 00 00    	je     4019ee <_Z12write_sparseP8_IO_FILEiiPKf+0x2fe>
  4017b5:	41 0f 2e 4e 10       	ucomiss 0x10(%r14),%xmm1
  4017ba:	7a 06                	jp     4017c2 <_Z12write_sparseP8_IO_FILEiiPKf+0xd2>
  4017bc:	0f 84 3e 02 00 00    	je     401a00 <_Z12write_sparseP8_IO_FILEiiPKf+0x310>
  4017c2:	ba 01 00 00 00       	mov    $0x1,%edx
  4017c7:	41 01 d7             	add    %edx,%r15d
  4017ca:	83 f8 05             	cmp    $0x5,%eax
  4017cd:	0f 84 3b 02 00 00    	je     401a0e <_Z12write_sparseP8_IO_FILEiiPKf+0x31e>
  4017d3:	41 0f 2e 4e 14       	ucomiss 0x14(%r14),%xmm1
  4017d8:	7a 06                	jp     4017e0 <_Z12write_sparseP8_IO_FILEiiPKf+0xf0>
  4017da:	0f 84 40 02 00 00    	je     401a20 <_Z12write_sparseP8_IO_FILEiiPKf+0x330>
  4017e0:	ba 01 00 00 00       	mov    $0x1,%edx
  4017e5:	41 01 d7             	add    %edx,%r15d
  4017e8:	b9 06 00 00 00       	mov    $0x6,%ecx
  4017ed:	41 39 c5             	cmp    %eax,%r13d
  4017f0:	0f 84 c7 00 00 00    	je     4018bd <_Z12write_sparseP8_IO_FILEiiPKf+0x1cd>
  4017f6:	44 89 ef             	mov    %r13d,%edi
  4017f9:	41 8d 75 ff          	lea    -0x1(%r13),%esi
  4017fd:	41 89 c1             	mov    %eax,%r9d
  401800:	29 c7                	sub    %eax,%edi
  401802:	8d 57 fc             	lea    -0x4(%rdi),%edx
  401805:	29 c6                	sub    %eax,%esi
  401807:	c1 ea 02             	shr    $0x2,%edx
  40180a:	83 c2 01             	add    $0x1,%edx
  40180d:	83 fe 02             	cmp    $0x2,%esi
  401810:	44 8d 04 95 00 00 00 	lea    0x0(,%rdx,4),%r8d
  401817:	00 
  401818:	76 4d                	jbe    401867 <_Z12write_sparseP8_IO_FILEiiPKf+0x177>
  40181a:	66 0f ef c0          	pxor   %xmm0,%xmm0
  40181e:	4b 8d 34 8e          	lea    (%r14,%r9,4),%rsi
  401822:	31 c0                	xor    %eax,%eax
  401824:	66 0f ef db          	pxor   %xmm3,%xmm3
  401828:	0f 28 16             	movaps (%rsi),%xmm2
  40182b:	83 c0 01             	add    $0x1,%eax
  40182e:	48 83 c6 10          	add    $0x10,%rsi
  401832:	39 c2                	cmp    %eax,%edx
  401834:	0f c2 d3 04          	cmpneqps %xmm3,%xmm2
  401838:	66 0f fa c2          	psubd  %xmm2,%xmm0
  40183c:	77 ea                	ja     401828 <_Z12write_sparseP8_IO_FILEiiPKf+0x138>
  40183e:	66 0f 6f d0          	movdqa %xmm0,%xmm2
  401842:	44 01 c1             	add    %r8d,%ecx
  401845:	66 0f 73 da 08       	psrldq $0x8,%xmm2
  40184a:	66 0f fe c2          	paddd  %xmm2,%xmm0
  40184e:	66 0f 6f d0          	movdqa %xmm0,%xmm2
  401852:	66 0f 73 da 04       	psrldq $0x4,%xmm2
  401857:	66 0f fe c2          	paddd  %xmm2,%xmm0
  40185b:	66 0f 7e c2          	movd   %xmm0,%edx
  40185f:	41 01 d7             	add    %edx,%r15d
  401862:	44 39 c7             	cmp    %r8d,%edi
  401865:	74 56                	je     4018bd <_Z12write_sparseP8_IO_FILEiiPKf+0x1cd>
  401867:	48 63 c1             	movslq %ecx,%rax
  40186a:	41 0f 2e 0c 86       	ucomiss (%r14,%rax,4),%xmm1
  40186f:	7a 06                	jp     401877 <_Z12write_sparseP8_IO_FILEiiPKf+0x187>
  401871:	0f 84 b9 01 00 00    	je     401a30 <_Z12write_sparseP8_IO_FILEiiPKf+0x340>
  401877:	b8 01 00 00 00       	mov    $0x1,%eax
  40187c:	41 01 c7             	add    %eax,%r15d
  40187f:	8d 41 01             	lea    0x1(%rcx),%eax
  401882:	41 39 c5             	cmp    %eax,%r13d
  401885:	7e 36                	jle    4018bd <_Z12write_sparseP8_IO_FILEiiPKf+0x1cd>
  401887:	48 98                	cltq   
  401889:	41 0f 2e 0c 86       	ucomiss (%r14,%rax,4),%xmm1
  40188e:	7a 06                	jp     401896 <_Z12write_sparseP8_IO_FILEiiPKf+0x1a6>
  401890:	0f 84 aa 01 00 00    	je     401a40 <_Z12write_sparseP8_IO_FILEiiPKf+0x350>
  401896:	b8 01 00 00 00       	mov    $0x1,%eax
  40189b:	41 01 c7             	add    %eax,%r15d
  40189e:	8d 41 02             	lea    0x2(%rcx),%eax
  4018a1:	41 39 c5             	cmp    %eax,%r13d
  4018a4:	7e 17                	jle    4018bd <_Z12write_sparseP8_IO_FILEiiPKf+0x1cd>
  4018a6:	48 98                	cltq   
  4018a8:	41 0f 2e 0c 86       	ucomiss (%r14,%rax,4),%xmm1
  4018ad:	7a 06                	jp     4018b5 <_Z12write_sparseP8_IO_FILEiiPKf+0x1c5>
  4018af:	0f 84 9b 01 00 00    	je     401a50 <_Z12write_sparseP8_IO_FILEiiPKf+0x360>
  4018b5:	b8 01 00 00 00       	mov    $0x1,%eax
  4018ba:	41 01 c7             	add    %eax,%r15d
  4018bd:	48 8d 74 24 10       	lea    0x10(%rsp),%rsi
  4018c2:	4c 89 e7             	mov    %r12,%rdi
  4018c5:	f3 0f 11 4c 24 0c    	movss  %xmm1,0xc(%rsp)
  4018cb:	c6 44 24 13 47       	movb   $0x47,0x13(%rsp)
  4018d0:	c6 44 24 10 4d       	movb   $0x4d,0x10(%rsp)
  4018d5:	c6 44 24 11 43       	movb   $0x43,0x11(%rsp)
  4018da:	c6 44 24 12 52       	movb   $0x52,0x12(%rsp)
  4018df:	e8 0c 12 00 00       	callq  402af0 <_Z15mm_write_bannerP8_IO_FILEPc>
  4018e4:	89 de                	mov    %ebx,%esi
  4018e6:	44 89 f9             	mov    %r15d,%ecx
  4018e9:	89 ea                	mov    %ebp,%edx
  4018eb:	4c 89 e7             	mov    %r12,%rdi
  4018ee:	31 db                	xor    %ebx,%ebx
  4018f0:	e8 4b 07 00 00       	callq  402040 <_Z21mm_write_mtx_crd_sizeP8_IO_FILEiii>
  4018f5:	f3 0f 10 4c 24 0c    	movss  0xc(%rsp),%xmm1
  4018fb:	eb 0c                	jmp    401909 <_Z12write_sparseP8_IO_FILEiiPKf+0x219>
  4018fd:	0f 1f 00             	nopl   (%rax)
  401900:	48 83 c3 01          	add    $0x1,%rbx
  401904:	41 39 dd             	cmp    %ebx,%r13d
  401907:	7e 43                	jle    40194c <_Z12write_sparseP8_IO_FILEiiPKf+0x25c>
  401909:	f3 41 0f 10 04 9e    	movss  (%r14,%rbx,4),%xmm0
  40190f:	89 d8                	mov    %ebx,%eax
  401911:	0f 2e c1             	ucomiss %xmm1,%xmm0
  401914:	7a 02                	jp     401918 <_Z12write_sparseP8_IO_FILEiiPKf+0x228>
  401916:	74 e8                	je     401900 <_Z12write_sparseP8_IO_FILEiiPKf+0x210>
  401918:	99                   	cltd   
  401919:	f3 0f 5a c0          	cvtss2sd %xmm0,%xmm0
  40191d:	be e2 2d 40 00       	mov    $0x402de2,%esi
  401922:	f7 fd                	idiv   %ebp
  401924:	4c 89 e7             	mov    %r12,%rdi
  401927:	48 83 c3 01          	add    $0x1,%rbx
  40192b:	f3 0f 11 4c 24 0c    	movss  %xmm1,0xc(%rsp)
  401931:	8d 4a 01             	lea    0x1(%rdx),%ecx
  401934:	8d 50 01             	lea    0x1(%rax),%edx
  401937:	b8 01 00 00 00       	mov    $0x1,%eax
  40193c:	e8 ff f1 ff ff       	callq  400b40 <fprintf@plt>
  401941:	41 39 dd             	cmp    %ebx,%r13d
  401944:	f3 0f 10 4c 24 0c    	movss  0xc(%rsp),%xmm1
  40194a:	7f bd                	jg     401909 <_Z12write_sparseP8_IO_FILEiiPKf+0x219>
  40194c:	48 83 c4 28          	add    $0x28,%rsp
  401950:	5b                   	pop    %rbx
  401951:	5d                   	pop    %rbp
  401952:	41 5c                	pop    %r12
  401954:	41 5d                	pop    %r13
  401956:	41 5e                	pop    %r14
  401958:	41 5f                	pop    %r15
  40195a:	c3                   	retq   
  40195b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  401960:	31 d2                	xor    %edx,%edx
  401962:	83 f8 01             	cmp    $0x1,%eax
  401965:	41 89 d7             	mov    %edx,%r15d
  401968:	0f 85 ec fd ff ff    	jne    40175a <_Z12write_sparseP8_IO_FILEiiPKf+0x6a>
  40196e:	b9 01 00 00 00       	mov    $0x1,%ecx
  401973:	e9 75 fe ff ff       	jmpq   4017ed <_Z12write_sparseP8_IO_FILEiiPKf+0xfd>
  401978:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  40197f:	00 
  401980:	85 c0                	test   %eax,%eax
  401982:	0f 85 b1 fd ff ff    	jne    401739 <_Z12write_sparseP8_IO_FILEiiPKf+0x49>
  401988:	45 31 ff             	xor    %r15d,%r15d
  40198b:	31 c9                	xor    %ecx,%ecx
  40198d:	66 0f ef c9          	pxor   %xmm1,%xmm1
  401991:	e9 60 fe ff ff       	jmpq   4017f6 <_Z12write_sparseP8_IO_FILEiiPKf+0x106>
  401996:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40199d:	00 00 00 
  4019a0:	45 31 ff             	xor    %r15d,%r15d
  4019a3:	41 01 d7             	add    %edx,%r15d
  4019a6:	83 f8 02             	cmp    $0x2,%eax
  4019a9:	0f 85 ca fd ff ff    	jne    401779 <_Z12write_sparseP8_IO_FILEiiPKf+0x89>
  4019af:	b9 02 00 00 00       	mov    $0x2,%ecx
  4019b4:	e9 34 fe ff ff       	jmpq   4017ed <_Z12write_sparseP8_IO_FILEiiPKf+0xfd>
  4019b9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  4019c0:	31 d2                	xor    %edx,%edx
  4019c2:	41 01 d7             	add    %edx,%r15d
  4019c5:	83 f8 03             	cmp    $0x3,%eax
  4019c8:	0f 85 c9 fd ff ff    	jne    401797 <_Z12write_sparseP8_IO_FILEiiPKf+0xa7>
  4019ce:	b9 03 00 00 00       	mov    $0x3,%ecx
  4019d3:	e9 15 fe ff ff       	jmpq   4017ed <_Z12write_sparseP8_IO_FILEiiPKf+0xfd>
  4019d8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  4019df:	00 
  4019e0:	31 d2                	xor    %edx,%edx
  4019e2:	41 01 d7             	add    %edx,%r15d
  4019e5:	83 f8 04             	cmp    $0x4,%eax
  4019e8:	0f 85 c7 fd ff ff    	jne    4017b5 <_Z12write_sparseP8_IO_FILEiiPKf+0xc5>
  4019ee:	b9 04 00 00 00       	mov    $0x4,%ecx
  4019f3:	e9 f5 fd ff ff       	jmpq   4017ed <_Z12write_sparseP8_IO_FILEiiPKf+0xfd>
  4019f8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  4019ff:	00 
  401a00:	31 d2                	xor    %edx,%edx
  401a02:	41 01 d7             	add    %edx,%r15d
  401a05:	83 f8 05             	cmp    $0x5,%eax
  401a08:	0f 85 c5 fd ff ff    	jne    4017d3 <_Z12write_sparseP8_IO_FILEiiPKf+0xe3>
  401a0e:	b9 05 00 00 00       	mov    $0x5,%ecx
  401a13:	e9 d5 fd ff ff       	jmpq   4017ed <_Z12write_sparseP8_IO_FILEiiPKf+0xfd>
  401a18:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  401a1f:	00 
  401a20:	31 d2                	xor    %edx,%edx
  401a22:	e9 be fd ff ff       	jmpq   4017e5 <_Z12write_sparseP8_IO_FILEiiPKf+0xf5>
  401a27:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  401a2e:	00 00 
  401a30:	31 c0                	xor    %eax,%eax
  401a32:	e9 45 fe ff ff       	jmpq   40187c <_Z12write_sparseP8_IO_FILEiiPKf+0x18c>
  401a37:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  401a3e:	00 00 
  401a40:	31 c0                	xor    %eax,%eax
  401a42:	e9 54 fe ff ff       	jmpq   40189b <_Z12write_sparseP8_IO_FILEiiPKf+0x1ab>
  401a47:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  401a4e:	00 00 
  401a50:	31 c0                	xor    %eax,%eax
  401a52:	e9 63 fe ff ff       	jmpq   4018ba <_Z12write_sparseP8_IO_FILEiiPKf+0x1ca>
  401a57:	48 8d 74 24 10       	lea    0x10(%rsp),%rsi
  401a5c:	c6 44 24 13 47       	movb   $0x47,0x13(%rsp)
  401a61:	c6 44 24 10 4d       	movb   $0x4d,0x10(%rsp)
  401a66:	c6 44 24 11 43       	movb   $0x43,0x11(%rsp)
  401a6b:	c6 44 24 12 52       	movb   $0x52,0x12(%rsp)
  401a70:	e8 7b 10 00 00       	callq  402af0 <_Z15mm_write_bannerP8_IO_FILEPc>
  401a75:	31 c9                	xor    %ecx,%ecx
  401a77:	89 ea                	mov    %ebp,%edx
  401a79:	89 de                	mov    %ebx,%esi
  401a7b:	4c 89 e7             	mov    %r12,%rdi
  401a7e:	e8 bd 05 00 00       	callq  402040 <_Z21mm_write_mtx_crd_sizeP8_IO_FILEiii>
  401a83:	e9 c4 fe ff ff       	jmpq   40194c <_Z12write_sparseP8_IO_FILEiiPKf+0x25c>
  401a88:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  401a8f:	00 

0000000000401a90 <_Z10read_denseP8_IO_FILEiiPf>:
  401a90:	85 f6                	test   %esi,%esi
  401a92:	0f 8e 80 00 00 00    	jle    401b18 <_Z10read_denseP8_IO_FILEiiPf+0x88>
  401a98:	85 d2                	test   %edx,%edx
  401a9a:	7e 7c                	jle    401b18 <_Z10read_denseP8_IO_FILEiiPf+0x88>
  401a9c:	41 57                	push   %r15
  401a9e:	41 56                	push   %r14
  401aa0:	48 63 c2             	movslq %edx,%rax
  401aa3:	41 55                	push   %r13
  401aa5:	41 54                	push   %r12
  401aa7:	48 c1 e0 02          	shl    $0x2,%rax
  401aab:	55                   	push   %rbp
  401aac:	53                   	push   %rbx
  401aad:	49 89 cc             	mov    %rcx,%r12
  401ab0:	48 89 fd             	mov    %rdi,%rbp
  401ab3:	45 31 ed             	xor    %r13d,%r13d
  401ab6:	48 83 ec 18          	sub    $0x18,%rsp
  401aba:	48 89 04 24          	mov    %rax,(%rsp)
  401abe:	8d 42 ff             	lea    -0x1(%rdx),%eax
  401ac1:	89 74 24 0c          	mov    %esi,0xc(%rsp)
  401ac5:	4c 8d 3c 85 04 00 00 	lea    0x4(,%rax,4),%r15
  401acc:	00 
  401acd:	0f 1f 00             	nopl   (%rax)
  401ad0:	4b 8d 1c 27          	lea    (%r15,%r12,1),%rbx
  401ad4:	4d 89 e6             	mov    %r12,%r14
  401ad7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  401ade:	00 00 
  401ae0:	4c 89 f2             	mov    %r14,%rdx
  401ae3:	31 c0                	xor    %eax,%eax
  401ae5:	be ec 2d 40 00       	mov    $0x402dec,%esi
  401aea:	48 89 ef             	mov    %rbp,%rdi
  401aed:	49 83 c6 04          	add    $0x4,%r14
  401af1:	e8 ba f0 ff ff       	callq  400bb0 <fscanf@plt>
  401af6:	49 39 de             	cmp    %rbx,%r14
  401af9:	75 e5                	jne    401ae0 <_Z10read_denseP8_IO_FILEiiPf+0x50>
  401afb:	41 83 c5 01          	add    $0x1,%r13d
  401aff:	4c 03 24 24          	add    (%rsp),%r12
  401b03:	44 39 6c 24 0c       	cmp    %r13d,0xc(%rsp)
  401b08:	75 c6                	jne    401ad0 <_Z10read_denseP8_IO_FILEiiPf+0x40>
  401b0a:	48 83 c4 18          	add    $0x18,%rsp
  401b0e:	5b                   	pop    %rbx
  401b0f:	5d                   	pop    %rbp
  401b10:	41 5c                	pop    %r12
  401b12:	41 5d                	pop    %r13
  401b14:	41 5e                	pop    %r14
  401b16:	41 5f                	pop    %r15
  401b18:	f3 c3                	repz retq 
  401b1a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000401b20 <_Z8read_matPiS_S_S_S_P8_IO_FILES1_>:
  401b20:	41 57                	push   %r15
  401b22:	41 56                	push   %r14
  401b24:	4d 89 c7             	mov    %r8,%r15
  401b27:	41 55                	push   %r13
  401b29:	41 54                	push   %r12
  401b2b:	49 89 fc             	mov    %rdi,%r12
  401b2e:	55                   	push   %rbp
  401b2f:	53                   	push   %rbx
  401b30:	48 89 f5             	mov    %rsi,%rbp
  401b33:	4c 89 cf             	mov    %r9,%rdi
  401b36:	49 89 cd             	mov    %rcx,%r13
  401b39:	4c 89 cb             	mov    %r9,%rbx
  401b3c:	48 83 ec 38          	sub    $0x38,%rsp
  401b40:	48 8d 74 24 10       	lea    0x10(%rsp),%rsi
  401b45:	48 89 54 24 08       	mov    %rdx,0x8(%rsp)
  401b4a:	e8 21 02 00 00       	callq  401d70 <_Z14mm_read_bannerP8_IO_FILEPA4_c>
  401b4f:	85 c0                	test   %eax,%eax
  401b51:	0f 85 f1 00 00 00    	jne    401c48 <_Z8read_matPiS_S_S_S_P8_IO_FILES1_+0x128>
  401b57:	48 8b 7c 24 70       	mov    0x70(%rsp),%rdi
  401b5c:	48 8d 74 24 20       	lea    0x20(%rsp),%rsi
  401b61:	e8 0a 02 00 00       	callq  401d70 <_Z14mm_read_bannerP8_IO_FILEPA4_c>
  401b66:	85 c0                	test   %eax,%eax
  401b68:	41 89 c6             	mov    %eax,%r14d
  401b6b:	0f 85 3f 01 00 00    	jne    401cb0 <_Z8read_matPiS_S_S_S_P8_IO_FILES1_+0x190>
  401b71:	80 7c 24 12 43       	cmpb   $0x43,0x12(%rsp)
  401b76:	0f 84 4c 01 00 00    	je     401cc8 <_Z8read_matPiS_S_S_S_P8_IO_FILES1_+0x1a8>
  401b7c:	80 7c 24 22 43       	cmpb   $0x43,0x22(%rsp)
  401b81:	0f 84 51 01 00 00    	je     401cd8 <_Z8read_matPiS_S_S_S_P8_IO_FILES1_+0x1b8>
  401b87:	80 7c 24 10 4d       	cmpb   $0x4d,0x10(%rsp)
  401b8c:	74 22                	je     401bb0 <_Z8read_matPiS_S_S_S_P8_IO_FILES1_+0x90>
  401b8e:	41 be f8 ff ff ff    	mov    $0xfffffff8,%r14d
  401b94:	48 83 c4 38          	add    $0x38,%rsp
  401b98:	44 89 f0             	mov    %r14d,%eax
  401b9b:	5b                   	pop    %rbx
  401b9c:	5d                   	pop    %rbp
  401b9d:	41 5c                	pop    %r12
  401b9f:	41 5d                	pop    %r13
  401ba1:	41 5e                	pop    %r14
  401ba3:	41 5f                	pop    %r15
  401ba5:	c3                   	retq   
  401ba6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  401bad:	00 00 00 
  401bb0:	0f b6 44 24 11       	movzbl 0x11(%rsp),%eax
  401bb5:	3c 43                	cmp    $0x43,%al
  401bb7:	0f 84 a3 00 00 00    	je     401c60 <_Z8read_matPiS_S_S_S_P8_IO_FILES1_+0x140>
  401bbd:	3c 41                	cmp    $0x41,%al
  401bbf:	75 cd                	jne    401b8e <_Z8read_matPiS_S_S_S_P8_IO_FILES1_+0x6e>
  401bc1:	41 c7 45 00 00 00 00 	movl   $0x0,0x0(%r13)
  401bc8:	00 
  401bc9:	48 89 ea             	mov    %rbp,%rdx
  401bcc:	4c 89 e6             	mov    %r12,%rsi
  401bcf:	48 89 df             	mov    %rbx,%rdi
  401bd2:	e8 59 05 00 00       	callq  402130 <_Z22mm_read_mtx_array_sizeP8_IO_FILEPiS1_>
  401bd7:	85 c0                	test   %eax,%eax
  401bd9:	75 5d                	jne    401c38 <_Z8read_matPiS_S_S_S_P8_IO_FILES1_+0x118>
  401bdb:	80 7c 24 20 4d       	cmpb   $0x4d,0x20(%rsp)
  401be0:	74 0e                	je     401bf0 <_Z8read_matPiS_S_S_S_P8_IO_FILES1_+0xd0>
  401be2:	41 be f7 ff ff ff    	mov    $0xfffffff7,%r14d
  401be8:	eb aa                	jmp    401b94 <_Z8read_matPiS_S_S_S_P8_IO_FILES1_+0x74>
  401bea:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  401bf0:	0f b6 44 24 21       	movzbl 0x21(%rsp),%eax
  401bf5:	3c 43                	cmp    $0x43,%al
  401bf7:	0f 84 8b 00 00 00    	je     401c88 <_Z8read_matPiS_S_S_S_P8_IO_FILES1_+0x168>
  401bfd:	3c 41                	cmp    $0x41,%al
  401bff:	75 e1                	jne    401be2 <_Z8read_matPiS_S_S_S_P8_IO_FILES1_+0xc2>
  401c01:	48 8b 54 24 08       	mov    0x8(%rsp),%rdx
  401c06:	48 8b 7c 24 70       	mov    0x70(%rsp),%rdi
  401c0b:	48 8d 74 24 2c       	lea    0x2c(%rsp),%rsi
  401c10:	41 c7 07 00 00 00 00 	movl   $0x0,(%r15)
  401c17:	e8 14 05 00 00       	callq  402130 <_Z22mm_read_mtx_array_sizeP8_IO_FILEPiS1_>
  401c1c:	85 c0                	test   %eax,%eax
  401c1e:	75 18                	jne    401c38 <_Z8read_matPiS_S_S_S_P8_IO_FILES1_+0x118>
  401c20:	8b 44 24 2c          	mov    0x2c(%rsp),%eax
  401c24:	39 45 00             	cmp    %eax,0x0(%rbp)
  401c27:	b8 f1 ff ff ff       	mov    $0xfffffff1,%eax
  401c2c:	44 0f 45 f0          	cmovne %eax,%r14d
  401c30:	e9 5f ff ff ff       	jmpq   401b94 <_Z8read_matPiS_S_S_S_P8_IO_FILES1_+0x74>
  401c35:	0f 1f 00             	nopl   (%rax)
  401c38:	41 be f5 ff ff ff    	mov    $0xfffffff5,%r14d
  401c3e:	e9 51 ff ff ff       	jmpq   401b94 <_Z8read_matPiS_S_S_S_P8_IO_FILES1_+0x74>
  401c43:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  401c48:	bf 50 2e 40 00       	mov    $0x402e50,%edi
  401c4d:	41 be fd ff ff ff    	mov    $0xfffffffd,%r14d
  401c53:	e8 08 ef ff ff       	callq  400b60 <puts@plt>
  401c58:	e9 37 ff ff ff       	jmpq   401b94 <_Z8read_matPiS_S_S_S_P8_IO_FILES1_+0x74>
  401c5d:	0f 1f 00             	nopl   (%rax)
  401c60:	4c 89 e9             	mov    %r13,%rcx
  401c63:	48 89 ea             	mov    %rbp,%rdx
  401c66:	4c 89 e6             	mov    %r12,%rsi
  401c69:	48 89 df             	mov    %rbx,%rdi
  401c6c:	e8 ff 03 00 00       	callq  402070 <_Z20mm_read_mtx_crd_sizeP8_IO_FILEPiS1_S1_>
  401c71:	85 c0                	test   %eax,%eax
  401c73:	0f 84 62 ff ff ff    	je     401bdb <_Z8read_matPiS_S_S_S_P8_IO_FILES1_+0xbb>
  401c79:	41 be f6 ff ff ff    	mov    $0xfffffff6,%r14d
  401c7f:	e9 10 ff ff ff       	jmpq   401b94 <_Z8read_matPiS_S_S_S_P8_IO_FILES1_+0x74>
  401c84:	0f 1f 40 00          	nopl   0x0(%rax)
  401c88:	48 8b 54 24 08       	mov    0x8(%rsp),%rdx
  401c8d:	48 8b 7c 24 70       	mov    0x70(%rsp),%rdi
  401c92:	48 8d 74 24 2c       	lea    0x2c(%rsp),%rsi
  401c97:	4c 89 f9             	mov    %r15,%rcx
  401c9a:	e8 d1 03 00 00       	callq  402070 <_Z20mm_read_mtx_crd_sizeP8_IO_FILEPiS1_S1_>
  401c9f:	85 c0                	test   %eax,%eax
  401ca1:	0f 84 79 ff ff ff    	je     401c20 <_Z8read_matPiS_S_S_S_P8_IO_FILES1_+0x100>
  401ca7:	eb d0                	jmp    401c79 <_Z8read_matPiS_S_S_S_P8_IO_FILES1_+0x159>
  401ca9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  401cb0:	bf 80 2e 40 00       	mov    $0x402e80,%edi
  401cb5:	41 be fc ff ff ff    	mov    $0xfffffffc,%r14d
  401cbb:	e8 a0 ee ff ff       	callq  400b60 <puts@plt>
  401cc0:	e9 cf fe ff ff       	jmpq   401b94 <_Z8read_matPiS_S_S_S_P8_IO_FILES1_+0x74>
  401cc5:	0f 1f 00             	nopl   (%rax)
  401cc8:	41 be fa ff ff ff    	mov    $0xfffffffa,%r14d
  401cce:	e9 c1 fe ff ff       	jmpq   401b94 <_Z8read_matPiS_S_S_S_P8_IO_FILES1_+0x74>
  401cd3:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  401cd8:	41 be f9 ff ff ff    	mov    $0xfffffff9,%r14d
  401cde:	e9 b1 fe ff ff       	jmpq   401b94 <_Z8read_matPiS_S_S_S_P8_IO_FILES1_+0x74>
  401ce3:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  401cea:	00 00 00 
  401ced:	0f 1f 00             	nopl   (%rax)

0000000000401cf0 <_Z11mm_is_validPc>:
  401cf0:	31 c0                	xor    %eax,%eax
  401cf2:	80 3f 4d             	cmpb   $0x4d,(%rdi)
  401cf5:	74 09                	je     401d00 <_Z11mm_is_validPc+0x10>
  401cf7:	f3 c3                	repz retq 
  401cf9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  401d00:	80 7f 01 41          	cmpb   $0x41,0x1(%rdi)
  401d04:	0f b6 57 02          	movzbl 0x2(%rdi),%edx
  401d08:	74 3e                	je     401d48 <_Z11mm_is_validPc+0x58>
  401d0a:	80 fa 52             	cmp    $0x52,%dl
  401d0d:	74 29                	je     401d38 <_Z11mm_is_validPc+0x48>
  401d0f:	80 fa 50             	cmp    $0x50,%dl
  401d12:	b8 01 00 00 00       	mov    $0x1,%eax
  401d17:	75 de                	jne    401cf7 <_Z11mm_is_validPc+0x7>
  401d19:	0f b6 57 03          	movzbl 0x3(%rdi),%edx
  401d1d:	80 fa 48             	cmp    $0x48,%dl
  401d20:	0f 94 c0             	sete   %al
  401d23:	80 fa 4b             	cmp    $0x4b,%dl
  401d26:	0f 94 c2             	sete   %dl
  401d29:	09 d0                	or     %edx,%eax
  401d2b:	83 f0 01             	xor    $0x1,%eax
  401d2e:	0f b6 c0             	movzbl %al,%eax
  401d31:	c3                   	retq   
  401d32:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  401d38:	31 c0                	xor    %eax,%eax
  401d3a:	80 7f 03 48          	cmpb   $0x48,0x3(%rdi)
  401d3e:	0f 95 c0             	setne  %al
  401d41:	c3                   	retq   
  401d42:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  401d48:	80 fa 50             	cmp    $0x50,%dl
  401d4b:	74 13                	je     401d60 <_Z11mm_is_validPc+0x70>
  401d4d:	80 fa 52             	cmp    $0x52,%dl
  401d50:	74 e6                	je     401d38 <_Z11mm_is_validPc+0x48>
  401d52:	b8 01 00 00 00       	mov    $0x1,%eax
  401d57:	c3                   	retq   
  401d58:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  401d5f:	00 
  401d60:	f3 c3                	repz retq 
  401d62:	0f 1f 40 00          	nopl   0x0(%rax)
  401d66:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  401d6d:	00 00 00 

0000000000401d70 <_Z14mm_read_bannerP8_IO_FILEPA4_c>:
  401d70:	41 57                	push   %r15
  401d72:	41 56                	push   %r14
  401d74:	48 89 fa             	mov    %rdi,%rdx
  401d77:	41 55                	push   %r13
  401d79:	41 54                	push   %r12
  401d7b:	55                   	push   %rbp
  401d7c:	53                   	push   %rbx
  401d7d:	48 89 f3             	mov    %rsi,%rbx
  401d80:	48 81 ec 58 05 00 00 	sub    $0x558,%rsp
  401d87:	c6 46 02 20          	movb   $0x20,0x2(%rsi)
  401d8b:	c6 46 01 20          	movb   $0x20,0x1(%rsi)
  401d8f:	48 8d bc 24 40 01 00 	lea    0x140(%rsp),%rdi
  401d96:	00 
  401d97:	c6 06 20             	movb   $0x20,(%rsi)
  401d9a:	c6 46 03 47          	movb   $0x47,0x3(%rsi)
  401d9e:	be 01 04 00 00       	mov    $0x401,%esi
  401da3:	e8 c8 ed ff ff       	callq  400b70 <fgets@plt>
  401da8:	48 85 c0             	test   %rax,%rax
  401dab:	74 4c                	je     401df9 <_Z14mm_read_bannerP8_IO_FILEPA4_c+0x89>
  401dad:	4c 8d bc 24 c0 00 00 	lea    0xc0(%rsp),%r15
  401db4:	00 
  401db5:	4c 8d b4 24 80 00 00 	lea    0x80(%rsp),%r14
  401dbc:	00 
  401dbd:	4c 8d 64 24 40       	lea    0x40(%rsp),%r12
  401dc2:	48 83 ec 08          	sub    $0x8,%rsp
  401dc6:	31 c0                	xor    %eax,%eax
  401dc8:	be 10 2f 40 00       	mov    $0x402f10,%esi
  401dcd:	4c 8d ac 24 08 01 00 	lea    0x108(%rsp),%r13
  401dd4:	00 
  401dd5:	4c 89 e1             	mov    %r12,%rcx
  401dd8:	4d 89 f9             	mov    %r15,%r9
  401ddb:	4d 89 f0             	mov    %r14,%r8
  401dde:	41 55                	push   %r13
  401de0:	48 8d 54 24 10       	lea    0x10(%rsp),%rdx
  401de5:	48 8d bc 24 50 01 00 	lea    0x150(%rsp),%rdi
  401dec:	00 
  401ded:	e8 ce ec ff ff       	callq  400ac0 <sscanf@plt>
  401df2:	83 f8 05             	cmp    $0x5,%eax
  401df5:	5a                   	pop    %rdx
  401df6:	59                   	pop    %rcx
  401df7:	74 17                	je     401e10 <_Z14mm_read_bannerP8_IO_FILEPA4_c+0xa0>
  401df9:	b8 0c 00 00 00       	mov    $0xc,%eax
  401dfe:	48 81 c4 58 05 00 00 	add    $0x558,%rsp
  401e05:	5b                   	pop    %rbx
  401e06:	5d                   	pop    %rbp
  401e07:	41 5c                	pop    %r12
  401e09:	41 5d                	pop    %r13
  401e0b:	41 5e                	pop    %r14
  401e0d:	41 5f                	pop    %r15
  401e0f:	c3                   	retq   
  401e10:	0f be 7c 24 40       	movsbl 0x40(%rsp),%edi
  401e15:	4c 89 e5             	mov    %r12,%rbp
  401e18:	40 84 ff             	test   %dil,%dil
  401e1b:	74 18                	je     401e35 <_Z14mm_read_bannerP8_IO_FILEPA4_c+0xc5>
  401e1d:	0f 1f 00             	nopl   (%rax)
  401e20:	e8 2b ed ff ff       	callq  400b50 <tolower@plt>
  401e25:	48 83 c5 01          	add    $0x1,%rbp
  401e29:	88 45 ff             	mov    %al,-0x1(%rbp)
  401e2c:	0f be 7d 00          	movsbl 0x0(%rbp),%edi
  401e30:	40 84 ff             	test   %dil,%dil
  401e33:	75 eb                	jne    401e20 <_Z14mm_read_bannerP8_IO_FILEPA4_c+0xb0>
  401e35:	0f be bc 24 80 00 00 	movsbl 0x80(%rsp),%edi
  401e3c:	00 
  401e3d:	40 84 ff             	test   %dil,%dil
  401e40:	74 1b                	je     401e5d <_Z14mm_read_bannerP8_IO_FILEPA4_c+0xed>
  401e42:	4c 89 f5             	mov    %r14,%rbp
  401e45:	0f 1f 00             	nopl   (%rax)
  401e48:	e8 03 ed ff ff       	callq  400b50 <tolower@plt>
  401e4d:	48 83 c5 01          	add    $0x1,%rbp
  401e51:	88 45 ff             	mov    %al,-0x1(%rbp)
  401e54:	0f be 7d 00          	movsbl 0x0(%rbp),%edi
  401e58:	40 84 ff             	test   %dil,%dil
  401e5b:	75 eb                	jne    401e48 <_Z14mm_read_bannerP8_IO_FILEPA4_c+0xd8>
  401e5d:	0f be bc 24 c0 00 00 	movsbl 0xc0(%rsp),%edi
  401e64:	00 
  401e65:	40 84 ff             	test   %dil,%dil
  401e68:	74 1b                	je     401e85 <_Z14mm_read_bannerP8_IO_FILEPA4_c+0x115>
  401e6a:	4c 89 fd             	mov    %r15,%rbp
  401e6d:	0f 1f 00             	nopl   (%rax)
  401e70:	e8 db ec ff ff       	callq  400b50 <tolower@plt>
  401e75:	48 83 c5 01          	add    $0x1,%rbp
  401e79:	88 45 ff             	mov    %al,-0x1(%rbp)
  401e7c:	0f be 7d 00          	movsbl 0x0(%rbp),%edi
  401e80:	40 84 ff             	test   %dil,%dil
  401e83:	75 eb                	jne    401e70 <_Z14mm_read_bannerP8_IO_FILEPA4_c+0x100>
  401e85:	0f be bc 24 00 01 00 	movsbl 0x100(%rsp),%edi
  401e8c:	00 
  401e8d:	4c 89 ed             	mov    %r13,%rbp
  401e90:	40 84 ff             	test   %dil,%dil
  401e93:	74 18                	je     401ead <_Z14mm_read_bannerP8_IO_FILEPA4_c+0x13d>
  401e95:	0f 1f 00             	nopl   (%rax)
  401e98:	e8 b3 ec ff ff       	callq  400b50 <tolower@plt>
  401e9d:	48 83 c5 01          	add    $0x1,%rbp
  401ea1:	88 45 ff             	mov    %al,-0x1(%rbp)
  401ea4:	0f be 7d 00          	movsbl 0x0(%rbp),%edi
  401ea8:	40 84 ff             	test   %dil,%dil
  401eab:	75 eb                	jne    401e98 <_Z14mm_read_bannerP8_IO_FILEPA4_c+0x128>
  401ead:	b9 0e 00 00 00       	mov    $0xe,%ecx
  401eb2:	bf 1f 2f 40 00       	mov    $0x402f1f,%edi
  401eb7:	48 89 e6             	mov    %rsp,%rsi
  401eba:	f3 a6                	repz cmpsb %es:(%rdi),%ds:(%rsi)
  401ebc:	b8 0e 00 00 00       	mov    $0xe,%eax
  401ec1:	0f 97 c1             	seta   %cl
  401ec4:	0f 92 c2             	setb   %dl
  401ec7:	38 d1                	cmp    %dl,%cl
  401ec9:	0f 85 2f ff ff ff    	jne    401dfe <_Z14mm_read_bannerP8_IO_FILEPA4_c+0x8e>
  401ecf:	bf 2e 2f 40 00       	mov    $0x402f2e,%edi
  401ed4:	b9 07 00 00 00       	mov    $0x7,%ecx
  401ed9:	4c 89 e6             	mov    %r12,%rsi
  401edc:	f3 a6                	repz cmpsb %es:(%rdi),%ds:(%rsi)
  401ede:	74 0a                	je     401eea <_Z14mm_read_bannerP8_IO_FILEPA4_c+0x17a>
  401ee0:	b8 0f 00 00 00       	mov    $0xf,%eax
  401ee5:	e9 14 ff ff ff       	jmpq   401dfe <_Z14mm_read_bannerP8_IO_FILEPA4_c+0x8e>
  401eea:	c6 03 4d             	movb   $0x4d,(%rbx)
  401eed:	bf 35 2f 40 00       	mov    $0x402f35,%edi
  401ef2:	b9 0b 00 00 00       	mov    $0xb,%ecx
  401ef7:	4c 89 f6             	mov    %r14,%rsi
  401efa:	f3 a6                	repz cmpsb %es:(%rdi),%ds:(%rsi)
  401efc:	0f 85 ae 00 00 00    	jne    401fb0 <_Z14mm_read_bannerP8_IO_FILEPA4_c+0x240>
  401f02:	c6 43 01 43          	movb   $0x43,0x1(%rbx)
  401f06:	bf 46 2f 40 00       	mov    $0x402f46,%edi
  401f0b:	b9 05 00 00 00       	mov    $0x5,%ecx
  401f10:	4c 89 fe             	mov    %r15,%rsi
  401f13:	f3 a6                	repz cmpsb %es:(%rdi),%ds:(%rsi)
  401f15:	0f 85 b3 00 00 00    	jne    401fce <_Z14mm_read_bannerP8_IO_FILEPA4_c+0x25e>
  401f1b:	c6 43 02 52          	movb   $0x52,0x2(%rbx)
  401f1f:	bf 63 2f 40 00       	mov    $0x402f63,%edi
  401f24:	b9 08 00 00 00       	mov    $0x8,%ecx
  401f29:	4c 89 ee             	mov    %r13,%rsi
  401f2c:	f3 a6                	repz cmpsb %es:(%rdi),%ds:(%rsi)
  401f2e:	0f 97 c0             	seta   %al
  401f31:	0f 92 c2             	setb   %dl
  401f34:	29 d0                	sub    %edx,%eax
  401f36:	0f be c0             	movsbl %al,%eax
  401f39:	85 c0                	test   %eax,%eax
  401f3b:	0f 84 a7 00 00 00    	je     401fe8 <_Z14mm_read_bannerP8_IO_FILEPA4_c+0x278>
  401f41:	bf 7a 2f 40 00       	mov    $0x402f7a,%edi
  401f46:	b9 0a 00 00 00       	mov    $0xa,%ecx
  401f4b:	4c 89 ee             	mov    %r13,%rsi
  401f4e:	f3 a6                	repz cmpsb %es:(%rdi),%ds:(%rsi)
  401f50:	0f 97 c0             	seta   %al
  401f53:	0f 92 c2             	setb   %dl
  401f56:	29 d0                	sub    %edx,%eax
  401f58:	0f be c0             	movsbl %al,%eax
  401f5b:	85 c0                	test   %eax,%eax
  401f5d:	0f 84 a8 00 00 00    	je     40200b <_Z14mm_read_bannerP8_IO_FILEPA4_c+0x29b>
  401f63:	bf 6b 2f 40 00       	mov    $0x402f6b,%edi
  401f68:	b9 0a 00 00 00       	mov    $0xa,%ecx
  401f6d:	4c 89 ee             	mov    %r13,%rsi
  401f70:	f3 a6                	repz cmpsb %es:(%rdi),%ds:(%rsi)
  401f72:	0f 97 c0             	seta   %al
  401f75:	0f 92 c2             	setb   %dl
  401f78:	29 d0                	sub    %edx,%eax
  401f7a:	0f be c0             	movsbl %al,%eax
  401f7d:	85 c0                	test   %eax,%eax
  401f7f:	0f 84 8f 00 00 00    	je     402014 <_Z14mm_read_bannerP8_IO_FILEPA4_c+0x2a4>
  401f85:	bf 75 2f 40 00       	mov    $0x402f75,%edi
  401f8a:	b9 0f 00 00 00       	mov    $0xf,%ecx
  401f8f:	4c 89 ee             	mov    %r13,%rsi
  401f92:	f3 a6                	repz cmpsb %es:(%rdi),%ds:(%rsi)
  401f94:	0f 97 c0             	seta   %al
  401f97:	0f 92 c2             	setb   %dl
  401f9a:	29 d0                	sub    %edx,%eax
  401f9c:	0f be c0             	movsbl %al,%eax
  401f9f:	85 c0                	test   %eax,%eax
  401fa1:	0f 85 39 ff ff ff    	jne    401ee0 <_Z14mm_read_bannerP8_IO_FILEPA4_c+0x170>
  401fa7:	c6 43 03 4b          	movb   $0x4b,0x3(%rbx)
  401fab:	e9 4e fe ff ff       	jmpq   401dfe <_Z14mm_read_bannerP8_IO_FILEPA4_c+0x8e>
  401fb0:	bf 40 2f 40 00       	mov    $0x402f40,%edi
  401fb5:	b9 06 00 00 00       	mov    $0x6,%ecx
  401fba:	4c 89 f6             	mov    %r14,%rsi
  401fbd:	f3 a6                	repz cmpsb %es:(%rdi),%ds:(%rsi)
  401fbf:	0f 85 1b ff ff ff    	jne    401ee0 <_Z14mm_read_bannerP8_IO_FILEPA4_c+0x170>
  401fc5:	c6 43 01 41          	movb   $0x41,0x1(%rbx)
  401fc9:	e9 38 ff ff ff       	jmpq   401f06 <_Z14mm_read_bannerP8_IO_FILEPA4_c+0x196>
  401fce:	bf 4b 2f 40 00       	mov    $0x402f4b,%edi
  401fd3:	b9 08 00 00 00       	mov    $0x8,%ecx
  401fd8:	4c 89 fe             	mov    %r15,%rsi
  401fdb:	f3 a6                	repz cmpsb %es:(%rdi),%ds:(%rsi)
  401fdd:	75 12                	jne    401ff1 <_Z14mm_read_bannerP8_IO_FILEPA4_c+0x281>
  401fdf:	c6 43 02 43          	movb   $0x43,0x2(%rbx)
  401fe3:	e9 37 ff ff ff       	jmpq   401f1f <_Z14mm_read_bannerP8_IO_FILEPA4_c+0x1af>
  401fe8:	c6 43 03 47          	movb   $0x47,0x3(%rbx)
  401fec:	e9 0d fe ff ff       	jmpq   401dfe <_Z14mm_read_bannerP8_IO_FILEPA4_c+0x8e>
  401ff1:	bf 53 2f 40 00       	mov    $0x402f53,%edi
  401ff6:	b9 08 00 00 00       	mov    $0x8,%ecx
  401ffb:	4c 89 fe             	mov    %r15,%rsi
  401ffe:	f3 a6                	repz cmpsb %es:(%rdi),%ds:(%rsi)
  402000:	75 1b                	jne    40201d <_Z14mm_read_bannerP8_IO_FILEPA4_c+0x2ad>
  402002:	c6 43 02 50          	movb   $0x50,0x2(%rbx)
  402006:	e9 14 ff ff ff       	jmpq   401f1f <_Z14mm_read_bannerP8_IO_FILEPA4_c+0x1af>
  40200b:	c6 43 03 53          	movb   $0x53,0x3(%rbx)
  40200f:	e9 ea fd ff ff       	jmpq   401dfe <_Z14mm_read_bannerP8_IO_FILEPA4_c+0x8e>
  402014:	c6 43 03 48          	movb   $0x48,0x3(%rbx)
  402018:	e9 e1 fd ff ff       	jmpq   401dfe <_Z14mm_read_bannerP8_IO_FILEPA4_c+0x8e>
  40201d:	be 5b 2f 40 00       	mov    $0x402f5b,%esi
  402022:	4c 89 ff             	mov    %r15,%rdi
  402025:	e8 06 eb ff ff       	callq  400b30 <strcmp@plt>
  40202a:	85 c0                	test   %eax,%eax
  40202c:	0f 85 ae fe ff ff    	jne    401ee0 <_Z14mm_read_bannerP8_IO_FILEPA4_c+0x170>
  402032:	c6 43 02 49          	movb   $0x49,0x2(%rbx)
  402036:	e9 e4 fe ff ff       	jmpq   401f1f <_Z14mm_read_bannerP8_IO_FILEPA4_c+0x1af>
  40203b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000402040 <_Z21mm_write_mtx_crd_sizeP8_IO_FILEiii>:
  402040:	48 83 ec 08          	sub    $0x8,%rsp
  402044:	41 89 c8             	mov    %ecx,%r8d
  402047:	31 c0                	xor    %eax,%eax
  402049:	89 d1                	mov    %edx,%ecx
  40204b:	89 f2                	mov    %esi,%edx
  40204d:	be 84 2f 40 00       	mov    $0x402f84,%esi
  402052:	e8 e9 ea ff ff       	callq  400b40 <fprintf@plt>
  402057:	ba 00 00 00 00       	mov    $0x0,%edx
  40205c:	83 f8 03             	cmp    $0x3,%eax
  40205f:	b8 11 00 00 00       	mov    $0x11,%eax
  402064:	0f 44 c2             	cmove  %edx,%eax
  402067:	48 83 c4 08          	add    $0x8,%rsp
  40206b:	c3                   	retq   
  40206c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000402070 <_Z20mm_read_mtx_crd_sizeP8_IO_FILEPiS1_S1_>:
  402070:	41 55                	push   %r13
  402072:	41 54                	push   %r12
  402074:	49 89 cd             	mov    %rcx,%r13
  402077:	55                   	push   %rbp
  402078:	53                   	push   %rbx
  402079:	48 89 f5             	mov    %rsi,%rbp
  40207c:	48 89 fb             	mov    %rdi,%rbx
  40207f:	49 89 d4             	mov    %rdx,%r12
  402082:	48 81 ec 18 04 00 00 	sub    $0x418,%rsp
  402089:	c7 01 00 00 00 00    	movl   $0x0,(%rcx)
  40208f:	c7 02 00 00 00 00    	movl   $0x0,(%rdx)
  402095:	c7 06 00 00 00 00    	movl   $0x0,(%rsi)
  40209b:	eb 09                	jmp    4020a6 <_Z20mm_read_mtx_crd_sizeP8_IO_FILEPiS1_S1_+0x36>
  40209d:	0f 1f 00             	nopl   (%rax)
  4020a0:	80 3c 24 25          	cmpb   $0x25,(%rsp)
  4020a4:	75 2a                	jne    4020d0 <_Z20mm_read_mtx_crd_sizeP8_IO_FILEPiS1_S1_+0x60>
  4020a6:	48 89 da             	mov    %rbx,%rdx
  4020a9:	be 01 04 00 00       	mov    $0x401,%esi
  4020ae:	48 89 e7             	mov    %rsp,%rdi
  4020b1:	e8 ba ea ff ff       	callq  400b70 <fgets@plt>
  4020b6:	48 85 c0             	test   %rax,%rax
  4020b9:	75 e5                	jne    4020a0 <_Z20mm_read_mtx_crd_sizeP8_IO_FILEPiS1_S1_+0x30>
  4020bb:	48 81 c4 18 04 00 00 	add    $0x418,%rsp
  4020c2:	b8 0c 00 00 00       	mov    $0xc,%eax
  4020c7:	5b                   	pop    %rbx
  4020c8:	5d                   	pop    %rbp
  4020c9:	41 5c                	pop    %r12
  4020cb:	41 5d                	pop    %r13
  4020cd:	c3                   	retq   
  4020ce:	66 90                	xchg   %ax,%ax
  4020d0:	4d 89 e8             	mov    %r13,%r8
  4020d3:	4c 89 e1             	mov    %r12,%rcx
  4020d6:	48 89 ea             	mov    %rbp,%rdx
  4020d9:	be 8e 2f 40 00       	mov    $0x402f8e,%esi
  4020de:	48 89 e7             	mov    %rsp,%rdi
  4020e1:	31 c0                	xor    %eax,%eax
  4020e3:	e8 d8 e9 ff ff       	callq  400ac0 <sscanf@plt>
  4020e8:	eb 23                	jmp    40210d <_Z20mm_read_mtx_crd_sizeP8_IO_FILEPiS1_S1_+0x9d>
  4020ea:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  4020f0:	31 c0                	xor    %eax,%eax
  4020f2:	4d 89 e8             	mov    %r13,%r8
  4020f5:	4c 89 e1             	mov    %r12,%rcx
  4020f8:	48 89 ea             	mov    %rbp,%rdx
  4020fb:	be 8e 2f 40 00       	mov    $0x402f8e,%esi
  402100:	48 89 df             	mov    %rbx,%rdi
  402103:	e8 a8 ea ff ff       	callq  400bb0 <fscanf@plt>
  402108:	83 f8 ff             	cmp    $0xffffffff,%eax
  40210b:	74 ae                	je     4020bb <_Z20mm_read_mtx_crd_sizeP8_IO_FILEPiS1_S1_+0x4b>
  40210d:	83 f8 03             	cmp    $0x3,%eax
  402110:	75 de                	jne    4020f0 <_Z20mm_read_mtx_crd_sizeP8_IO_FILEPiS1_S1_+0x80>
  402112:	48 81 c4 18 04 00 00 	add    $0x418,%rsp
  402119:	31 c0                	xor    %eax,%eax
  40211b:	5b                   	pop    %rbx
  40211c:	5d                   	pop    %rbp
  40211d:	41 5c                	pop    %r12
  40211f:	41 5d                	pop    %r13
  402121:	c3                   	retq   
  402122:	0f 1f 40 00          	nopl   0x0(%rax)
  402126:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40212d:	00 00 00 

0000000000402130 <_Z22mm_read_mtx_array_sizeP8_IO_FILEPiS1_>:
  402130:	41 54                	push   %r12
  402132:	55                   	push   %rbp
  402133:	49 89 d4             	mov    %rdx,%r12
  402136:	53                   	push   %rbx
  402137:	48 89 f5             	mov    %rsi,%rbp
  40213a:	48 89 fb             	mov    %rdi,%rbx
  40213d:	48 81 ec 10 04 00 00 	sub    $0x410,%rsp
  402144:	c7 02 00 00 00 00    	movl   $0x0,(%rdx)
  40214a:	c7 06 00 00 00 00    	movl   $0x0,(%rsi)
  402150:	eb 0c                	jmp    40215e <_Z22mm_read_mtx_array_sizeP8_IO_FILEPiS1_+0x2e>
  402152:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  402158:	80 3c 24 25          	cmpb   $0x25,(%rsp)
  40215c:	75 2a                	jne    402188 <_Z22mm_read_mtx_array_sizeP8_IO_FILEPiS1_+0x58>
  40215e:	48 89 da             	mov    %rbx,%rdx
  402161:	be 01 04 00 00       	mov    $0x401,%esi
  402166:	48 89 e7             	mov    %rsp,%rdi
  402169:	e8 02 ea ff ff       	callq  400b70 <fgets@plt>
  40216e:	48 85 c0             	test   %rax,%rax
  402171:	75 e5                	jne    402158 <_Z22mm_read_mtx_array_sizeP8_IO_FILEPiS1_+0x28>
  402173:	48 81 c4 10 04 00 00 	add    $0x410,%rsp
  40217a:	b8 0c 00 00 00       	mov    $0xc,%eax
  40217f:	5b                   	pop    %rbx
  402180:	5d                   	pop    %rbp
  402181:	41 5c                	pop    %r12
  402183:	c3                   	retq   
  402184:	0f 1f 40 00          	nopl   0x0(%rax)
  402188:	4c 89 e1             	mov    %r12,%rcx
  40218b:	48 89 ea             	mov    %rbp,%rdx
  40218e:	be 91 2f 40 00       	mov    $0x402f91,%esi
  402193:	48 89 e7             	mov    %rsp,%rdi
  402196:	31 c0                	xor    %eax,%eax
  402198:	e8 23 e9 ff ff       	callq  400ac0 <sscanf@plt>
  40219d:	eb 1b                	jmp    4021ba <_Z22mm_read_mtx_array_sizeP8_IO_FILEPiS1_+0x8a>
  40219f:	90                   	nop
  4021a0:	31 c0                	xor    %eax,%eax
  4021a2:	4c 89 e1             	mov    %r12,%rcx
  4021a5:	48 89 ea             	mov    %rbp,%rdx
  4021a8:	be 91 2f 40 00       	mov    $0x402f91,%esi
  4021ad:	48 89 df             	mov    %rbx,%rdi
  4021b0:	e8 fb e9 ff ff       	callq  400bb0 <fscanf@plt>
  4021b5:	83 f8 ff             	cmp    $0xffffffff,%eax
  4021b8:	74 b9                	je     402173 <_Z22mm_read_mtx_array_sizeP8_IO_FILEPiS1_+0x43>
  4021ba:	83 f8 02             	cmp    $0x2,%eax
  4021bd:	75 e1                	jne    4021a0 <_Z22mm_read_mtx_array_sizeP8_IO_FILEPiS1_+0x70>
  4021bf:	48 81 c4 10 04 00 00 	add    $0x410,%rsp
  4021c6:	31 c0                	xor    %eax,%eax
  4021c8:	5b                   	pop    %rbx
  4021c9:	5d                   	pop    %rbp
  4021ca:	41 5c                	pop    %r12
  4021cc:	c3                   	retq   
  4021cd:	0f 1f 00             	nopl   (%rax)

00000000004021d0 <_Z23mm_write_mtx_array_sizeP8_IO_FILEii>:
  4021d0:	48 83 ec 08          	sub    $0x8,%rsp
  4021d4:	89 d1                	mov    %edx,%ecx
  4021d6:	31 c0                	xor    %eax,%eax
  4021d8:	89 f2                	mov    %esi,%edx
  4021da:	be 87 2f 40 00       	mov    $0x402f87,%esi
  4021df:	e8 5c e9 ff ff       	callq  400b40 <fprintf@plt>
  4021e4:	ba 00 00 00 00       	mov    $0x0,%edx
  4021e9:	83 f8 02             	cmp    $0x2,%eax
  4021ec:	b8 11 00 00 00       	mov    $0x11,%eax
  4021f1:	0f 44 c2             	cmove  %edx,%eax
  4021f4:	48 83 c4 08          	add    $0x8,%rsp
  4021f8:	c3                   	retq   
  4021f9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000402200 <_Z20mm_read_mtx_crd_dataP8_IO_FILEiiiPiS1_PdPc>:
  402200:	41 56                	push   %r14
  402202:	41 55                	push   %r13
  402204:	41 54                	push   %r12
  402206:	55                   	push   %rbp
  402207:	49 89 fc             	mov    %rdi,%r12
  40220a:	53                   	push   %rbx
  40220b:	48 8b 44 24 38       	mov    0x38(%rsp),%rax
  402210:	0f b6 50 02          	movzbl 0x2(%rax),%edx
  402214:	80 fa 43             	cmp    $0x43,%dl
  402217:	74 27                	je     402240 <_Z20mm_read_mtx_crd_dataP8_IO_FILEiiiPiS1_PdPc+0x40>
  402219:	80 fa 52             	cmp    $0x52,%dl
  40221c:	0f 84 7e 00 00 00    	je     4022a0 <_Z20mm_read_mtx_crd_dataP8_IO_FILEiiiPiS1_PdPc+0xa0>
  402222:	80 fa 50             	cmp    $0x50,%dl
  402225:	b8 0f 00 00 00       	mov    $0xf,%eax
  40222a:	0f 84 d0 00 00 00    	je     402300 <_Z20mm_read_mtx_crd_dataP8_IO_FILEiiiPiS1_PdPc+0x100>
  402230:	5b                   	pop    %rbx
  402231:	5d                   	pop    %rbp
  402232:	41 5c                	pop    %r12
  402234:	41 5d                	pop    %r13
  402236:	41 5e                	pop    %r14
  402238:	c3                   	retq   
  402239:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  402240:	85 c9                	test   %ecx,%ecx
  402242:	7e 60                	jle    4022a4 <_Z20mm_read_mtx_crd_dataP8_IO_FILEiiiPiS1_PdPc+0xa4>
  402244:	48 8b 44 24 30       	mov    0x30(%rsp),%rax
  402249:	4c 89 c3             	mov    %r8,%rbx
  40224c:	4c 89 cd             	mov    %r9,%rbp
  40224f:	4c 8d 68 08          	lea    0x8(%rax),%r13
  402253:	8d 41 ff             	lea    -0x1(%rcx),%eax
  402256:	4d 8d 74 80 04       	lea    0x4(%r8,%rax,4),%r14
  40225b:	eb 14                	jmp    402271 <_Z20mm_read_mtx_crd_dataP8_IO_FILEiiiPiS1_PdPc+0x71>
  40225d:	0f 1f 00             	nopl   (%rax)
  402260:	48 83 c3 04          	add    $0x4,%rbx
  402264:	48 83 c5 04          	add    $0x4,%rbp
  402268:	49 83 c5 10          	add    $0x10,%r13
  40226c:	49 39 de             	cmp    %rbx,%r14
  40226f:	74 33                	je     4022a4 <_Z20mm_read_mtx_crd_dataP8_IO_FILEiiiPiS1_PdPc+0xa4>
  402271:	4d 8d 45 f8          	lea    -0x8(%r13),%r8
  402275:	31 c0                	xor    %eax,%eax
  402277:	4d 89 e9             	mov    %r13,%r9
  40227a:	48 89 e9             	mov    %rbp,%rcx
  40227d:	48 89 da             	mov    %rbx,%rdx
  402280:	be 97 2f 40 00       	mov    $0x402f97,%esi
  402285:	4c 89 e7             	mov    %r12,%rdi
  402288:	e8 23 e9 ff ff       	callq  400bb0 <fscanf@plt>
  40228d:	83 f8 04             	cmp    $0x4,%eax
  402290:	74 ce                	je     402260 <_Z20mm_read_mtx_crd_dataP8_IO_FILEiiiPiS1_PdPc+0x60>
  402292:	5b                   	pop    %rbx
  402293:	b8 0c 00 00 00       	mov    $0xc,%eax
  402298:	5d                   	pop    %rbp
  402299:	41 5c                	pop    %r12
  40229b:	41 5d                	pop    %r13
  40229d:	41 5e                	pop    %r14
  40229f:	c3                   	retq   
  4022a0:	85 c9                	test   %ecx,%ecx
  4022a2:	7f 0c                	jg     4022b0 <_Z20mm_read_mtx_crd_dataP8_IO_FILEiiiPiS1_PdPc+0xb0>
  4022a4:	5b                   	pop    %rbx
  4022a5:	31 c0                	xor    %eax,%eax
  4022a7:	5d                   	pop    %rbp
  4022a8:	41 5c                	pop    %r12
  4022aa:	41 5d                	pop    %r13
  4022ac:	41 5e                	pop    %r14
  4022ae:	c3                   	retq   
  4022af:	90                   	nop
  4022b0:	8d 41 ff             	lea    -0x1(%rcx),%eax
  4022b3:	4c 89 c3             	mov    %r8,%rbx
  4022b6:	4c 89 cd             	mov    %r9,%rbp
  4022b9:	4c 8b 6c 24 30       	mov    0x30(%rsp),%r13
  4022be:	4d 8d 74 80 04       	lea    0x4(%r8,%rax,4),%r14
  4022c3:	eb 14                	jmp    4022d9 <_Z20mm_read_mtx_crd_dataP8_IO_FILEiiiPiS1_PdPc+0xd9>
  4022c5:	0f 1f 00             	nopl   (%rax)
  4022c8:	48 83 c3 04          	add    $0x4,%rbx
  4022cc:	48 83 c5 04          	add    $0x4,%rbp
  4022d0:	49 83 c5 08          	add    $0x8,%r13
  4022d4:	4c 39 f3             	cmp    %r14,%rbx
  4022d7:	74 cb                	je     4022a4 <_Z20mm_read_mtx_crd_dataP8_IO_FILEiiiPiS1_PdPc+0xa4>
  4022d9:	31 c0                	xor    %eax,%eax
  4022db:	4d 89 e8             	mov    %r13,%r8
  4022de:	48 89 e9             	mov    %rbp,%rcx
  4022e1:	48 89 da             	mov    %rbx,%rdx
  4022e4:	be a5 2f 40 00       	mov    $0x402fa5,%esi
  4022e9:	4c 89 e7             	mov    %r12,%rdi
  4022ec:	e8 bf e8 ff ff       	callq  400bb0 <fscanf@plt>
  4022f1:	83 f8 03             	cmp    $0x3,%eax
  4022f4:	74 d2                	je     4022c8 <_Z20mm_read_mtx_crd_dataP8_IO_FILEiiiPiS1_PdPc+0xc8>
  4022f6:	eb 9a                	jmp    402292 <_Z20mm_read_mtx_crd_dataP8_IO_FILEiiiPiS1_PdPc+0x92>
  4022f8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  4022ff:	00 
  402300:	85 c9                	test   %ecx,%ecx
  402302:	7e a0                	jle    4022a4 <_Z20mm_read_mtx_crd_dataP8_IO_FILEiiiPiS1_PdPc+0xa4>
  402304:	8d 41 ff             	lea    -0x1(%rcx),%eax
  402307:	4c 89 c3             	mov    %r8,%rbx
  40230a:	4c 89 cd             	mov    %r9,%rbp
  40230d:	4d 8d 6c 80 04       	lea    0x4(%r8,%rax,4),%r13
  402312:	eb 15                	jmp    402329 <_Z20mm_read_mtx_crd_dataP8_IO_FILEiiiPiS1_PdPc+0x129>
  402314:	0f 1f 40 00          	nopl   0x0(%rax)
  402318:	48 83 c3 04          	add    $0x4,%rbx
  40231c:	48 83 c5 04          	add    $0x4,%rbp
  402320:	4c 39 eb             	cmp    %r13,%rbx
  402323:	0f 84 7b ff ff ff    	je     4022a4 <_Z20mm_read_mtx_crd_dataP8_IO_FILEiiiPiS1_PdPc+0xa4>
  402329:	31 c0                	xor    %eax,%eax
  40232b:	48 89 e9             	mov    %rbp,%rcx
  40232e:	48 89 da             	mov    %rbx,%rdx
  402331:	be 91 2f 40 00       	mov    $0x402f91,%esi
  402336:	4c 89 e7             	mov    %r12,%rdi
  402339:	e8 72 e8 ff ff       	callq  400bb0 <fscanf@plt>
  40233e:	83 f8 02             	cmp    $0x2,%eax
  402341:	74 d5                	je     402318 <_Z20mm_read_mtx_crd_dataP8_IO_FILEiiiPiS1_PdPc+0x118>
  402343:	e9 4a ff ff ff       	jmpq   402292 <_Z20mm_read_mtx_crd_dataP8_IO_FILEiiiPiS1_PdPc+0x92>
  402348:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  40234f:	00 

0000000000402350 <_Z21mm_read_mtx_crd_entryP8_IO_FILEPiS1_PdS2_Pc>:
  402350:	48 83 ec 08          	sub    $0x8,%rsp
  402354:	45 0f b6 49 02       	movzbl 0x2(%r9),%r9d
  402359:	41 80 f9 43          	cmp    $0x43,%r9b
  40235d:	74 19                	je     402378 <_Z21mm_read_mtx_crd_entryP8_IO_FILEPiS1_PdS2_Pc+0x28>
  40235f:	41 80 f9 52          	cmp    $0x52,%r9b
  402363:	74 3b                	je     4023a0 <_Z21mm_read_mtx_crd_entryP8_IO_FILEPiS1_PdS2_Pc+0x50>
  402365:	41 80 f9 50          	cmp    $0x50,%r9b
  402369:	b8 0f 00 00 00       	mov    $0xf,%eax
  40236e:	74 50                	je     4023c0 <_Z21mm_read_mtx_crd_entryP8_IO_FILEPiS1_PdS2_Pc+0x70>
  402370:	48 83 c4 08          	add    $0x8,%rsp
  402374:	c3                   	retq   
  402375:	0f 1f 00             	nopl   (%rax)
  402378:	4d 89 c1             	mov    %r8,%r9
  40237b:	31 c0                	xor    %eax,%eax
  40237d:	49 89 c8             	mov    %rcx,%r8
  402380:	48 89 d1             	mov    %rdx,%rcx
  402383:	48 89 f2             	mov    %rsi,%rdx
  402386:	be 97 2f 40 00       	mov    $0x402f97,%esi
  40238b:	e8 20 e8 ff ff       	callq  400bb0 <fscanf@plt>
  402390:	83 f8 04             	cmp    $0x4,%eax
  402393:	74 25                	je     4023ba <_Z21mm_read_mtx_crd_entryP8_IO_FILEPiS1_PdS2_Pc+0x6a>
  402395:	b8 0c 00 00 00       	mov    $0xc,%eax
  40239a:	48 83 c4 08          	add    $0x8,%rsp
  40239e:	c3                   	retq   
  40239f:	90                   	nop
  4023a0:	49 89 c8             	mov    %rcx,%r8
  4023a3:	31 c0                	xor    %eax,%eax
  4023a5:	48 89 d1             	mov    %rdx,%rcx
  4023a8:	48 89 f2             	mov    %rsi,%rdx
  4023ab:	be a5 2f 40 00       	mov    $0x402fa5,%esi
  4023b0:	e8 fb e7 ff ff       	callq  400bb0 <fscanf@plt>
  4023b5:	83 f8 03             	cmp    $0x3,%eax
  4023b8:	75 db                	jne    402395 <_Z21mm_read_mtx_crd_entryP8_IO_FILEPiS1_PdS2_Pc+0x45>
  4023ba:	31 c0                	xor    %eax,%eax
  4023bc:	eb b2                	jmp    402370 <_Z21mm_read_mtx_crd_entryP8_IO_FILEPiS1_PdS2_Pc+0x20>
  4023be:	66 90                	xchg   %ax,%ax
  4023c0:	48 89 d1             	mov    %rdx,%rcx
  4023c3:	31 c0                	xor    %eax,%eax
  4023c5:	48 89 f2             	mov    %rsi,%rdx
  4023c8:	be 91 2f 40 00       	mov    $0x402f91,%esi
  4023cd:	e8 de e7 ff ff       	callq  400bb0 <fscanf@plt>
  4023d2:	83 f8 02             	cmp    $0x2,%eax
  4023d5:	75 be                	jne    402395 <_Z21mm_read_mtx_crd_entryP8_IO_FILEPiS1_PdS2_Pc+0x45>
  4023d7:	31 c0                	xor    %eax,%eax
  4023d9:	eb 95                	jmp    402370 <_Z21mm_read_mtx_crd_entryP8_IO_FILEPiS1_PdS2_Pc+0x20>
  4023db:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

00000000004023e0 <_Z15mm_read_mtx_crdPcPiS0_S0_PS0_S1_PPdPA4_c>:
  4023e0:	41 57                	push   %r15
  4023e2:	41 56                	push   %r14
  4023e4:	48 89 f8             	mov    %rdi,%rax
  4023e7:	41 55                	push   %r13
  4023e9:	41 54                	push   %r12
  4023eb:	bf b0 2f 40 00       	mov    $0x402fb0,%edi
  4023f0:	55                   	push   %rbp
  4023f1:	53                   	push   %rbx
  4023f2:	49 89 f4             	mov    %rsi,%r12
  4023f5:	48 89 cd             	mov    %rcx,%rbp
  4023f8:	48 89 c6             	mov    %rax,%rsi
  4023fb:	b9 06 00 00 00       	mov    $0x6,%ecx
  402400:	48 83 ec 18          	sub    $0x18,%rsp
  402404:	49 89 d5             	mov    %rdx,%r13
  402407:	4d 89 c6             	mov    %r8,%r14
  40240a:	f3 a6                	repz cmpsb %es:(%rdi),%ds:(%rsi)
  40240c:	4d 89 cf             	mov    %r9,%r15
  40240f:	48 8b 1d d2 1c 20 00 	mov    0x201cd2(%rip),%rbx        # 6040e8 <stdin@@GLIBC_2.2.5>
  402416:	75 78                	jne    402490 <_Z15mm_read_mtx_crdPcPiS0_S0_PS0_S1_PPdPA4_c+0xb0>
  402418:	48 8b 74 24 58       	mov    0x58(%rsp),%rsi
  40241d:	48 89 df             	mov    %rbx,%rdi
  402420:	e8 4b f9 ff ff       	callq  401d70 <_Z14mm_read_bannerP8_IO_FILEPA4_c>
  402425:	85 c0                	test   %eax,%eax
  402427:	41 89 c2             	mov    %eax,%r10d
  40242a:	75 10                	jne    40243c <_Z15mm_read_mtx_crdPcPiS0_S0_PS0_S1_PPdPA4_c+0x5c>
  40242c:	48 8b 44 24 58       	mov    0x58(%rsp),%rax
  402431:	41 ba 0f 00 00 00    	mov    $0xf,%r10d
  402437:	80 38 4d             	cmpb   $0x4d,(%rax)
  40243a:	74 14                	je     402450 <_Z15mm_read_mtx_crdPcPiS0_S0_PS0_S1_PPdPA4_c+0x70>
  40243c:	48 83 c4 18          	add    $0x18,%rsp
  402440:	44 89 d0             	mov    %r10d,%eax
  402443:	5b                   	pop    %rbx
  402444:	5d                   	pop    %rbp
  402445:	41 5c                	pop    %r12
  402447:	41 5d                	pop    %r13
  402449:	41 5e                	pop    %r14
  40244b:	41 5f                	pop    %r15
  40244d:	c3                   	retq   
  40244e:	66 90                	xchg   %ax,%ax
  402450:	0f b6 40 01          	movzbl 0x1(%rax),%eax
  402454:	3c 41                	cmp    $0x41,%al
  402456:	74 e4                	je     40243c <_Z15mm_read_mtx_crdPcPiS0_S0_PS0_S1_PPdPA4_c+0x5c>
  402458:	48 8b 54 24 58       	mov    0x58(%rsp),%rdx
  40245d:	0f b6 52 02          	movzbl 0x2(%rdx),%edx
  402461:	80 fa 52             	cmp    $0x52,%dl
  402464:	74 52                	je     4024b8 <_Z15mm_read_mtx_crdPcPiS0_S0_PS0_S1_PPdPA4_c+0xd8>
  402466:	80 fa 50             	cmp    $0x50,%dl
  402469:	75 5c                	jne    4024c7 <_Z15mm_read_mtx_crdPcPiS0_S0_PS0_S1_PPdPA4_c+0xe7>
  40246b:	48 8b 54 24 58       	mov    0x58(%rsp),%rdx
  402470:	0f b6 52 03          	movzbl 0x3(%rdx),%edx
  402474:	80 fa 48             	cmp    $0x48,%dl
  402477:	74 05                	je     40247e <_Z15mm_read_mtx_crdPcPiS0_S0_PS0_S1_PPdPA4_c+0x9e>
  402479:	80 fa 4b             	cmp    $0x4b,%dl
  40247c:	75 49                	jne    4024c7 <_Z15mm_read_mtx_crdPcPiS0_S0_PS0_S1_PPdPA4_c+0xe7>
  40247e:	41 ba 0f 00 00 00    	mov    $0xf,%r10d
  402484:	eb b6                	jmp    40243c <_Z15mm_read_mtx_crdPcPiS0_S0_PS0_S1_PPdPA4_c+0x5c>
  402486:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40248d:	00 00 00 
  402490:	be 61 2f 40 00       	mov    $0x402f61,%esi
  402495:	48 89 c7             	mov    %rax,%rdi
  402498:	e8 53 e6 ff ff       	callq  400af0 <fopen@plt>
  40249d:	48 85 c0             	test   %rax,%rax
  4024a0:	48 89 c3             	mov    %rax,%rbx
  4024a3:	41 ba 0b 00 00 00    	mov    $0xb,%r10d
  4024a9:	0f 85 69 ff ff ff    	jne    402418 <_Z15mm_read_mtx_crdPcPiS0_S0_PS0_S1_PPdPA4_c+0x38>
  4024af:	eb 8b                	jmp    40243c <_Z15mm_read_mtx_crdPcPiS0_S0_PS0_S1_PPdPA4_c+0x5c>
  4024b1:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  4024b8:	48 8b 54 24 58       	mov    0x58(%rsp),%rdx
  4024bd:	80 7a 03 48          	cmpb   $0x48,0x3(%rdx)
  4024c1:	0f 84 75 ff ff ff    	je     40243c <_Z15mm_read_mtx_crdPcPiS0_S0_PS0_S1_PPdPA4_c+0x5c>
  4024c7:	3c 43                	cmp    $0x43,%al
  4024c9:	41 ba 0f 00 00 00    	mov    $0xf,%r10d
  4024cf:	0f 85 67 ff ff ff    	jne    40243c <_Z15mm_read_mtx_crdPcPiS0_S0_PS0_S1_PPdPA4_c+0x5c>
  4024d5:	48 89 e9             	mov    %rbp,%rcx
  4024d8:	4c 89 ea             	mov    %r13,%rdx
  4024db:	4c 89 e6             	mov    %r12,%rsi
  4024de:	48 89 df             	mov    %rbx,%rdi
  4024e1:	e8 8a fb ff ff       	callq  402070 <_Z20mm_read_mtx_crd_sizeP8_IO_FILEPiS1_S1_>
  4024e6:	85 c0                	test   %eax,%eax
  4024e8:	41 89 c2             	mov    %eax,%r10d
  4024eb:	0f 85 4b ff ff ff    	jne    40243c <_Z15mm_read_mtx_crdPcPiS0_S0_PS0_S1_PPdPA4_c+0x5c>
  4024f1:	48 63 7d 00          	movslq 0x0(%rbp),%rdi
  4024f5:	89 44 24 04          	mov    %eax,0x4(%rsp)
  4024f9:	48 c1 e7 02          	shl    $0x2,%rdi
  4024fd:	e8 1e e6 ff ff       	callq  400b20 <malloc@plt>
  402502:	48 63 7d 00          	movslq 0x0(%rbp),%rdi
  402506:	49 89 06             	mov    %rax,(%r14)
  402509:	48 c1 e7 02          	shl    $0x2,%rdi
  40250d:	e8 0e e6 ff ff       	callq  400b20 <malloc@plt>
  402512:	49 89 07             	mov    %rax,(%r15)
  402515:	49 89 c1             	mov    %rax,%r9
  402518:	48 8b 44 24 50       	mov    0x50(%rsp),%rax
  40251d:	44 8b 54 24 04       	mov    0x4(%rsp),%r10d
  402522:	48 c7 00 00 00 00 00 	movq   $0x0,(%rax)
  402529:	48 8b 44 24 58       	mov    0x58(%rsp),%rax
  40252e:	0f b6 40 02          	movzbl 0x2(%rax),%eax
  402532:	3c 43                	cmp    $0x43,%al
  402534:	74 30                	je     402566 <_Z15mm_read_mtx_crdPcPiS0_S0_PS0_S1_PPdPA4_c+0x186>
  402536:	3c 52                	cmp    $0x52,%al
  402538:	74 7e                	je     4025b8 <_Z15mm_read_mtx_crdPcPiS0_S0_PS0_S1_PPdPA4_c+0x1d8>
  40253a:	3c 50                	cmp    $0x50,%al
  40253c:	0f 84 86 00 00 00    	je     4025c8 <_Z15mm_read_mtx_crdPcPiS0_S0_PS0_S1_PPdPA4_c+0x1e8>
  402542:	48 3b 1d 9f 1b 20 00 	cmp    0x201b9f(%rip),%rbx        # 6040e8 <stdin@@GLIBC_2.2.5>
  402549:	0f 84 ed fe ff ff    	je     40243c <_Z15mm_read_mtx_crdPcPiS0_S0_PS0_S1_PPdPA4_c+0x5c>
  40254f:	48 89 df             	mov    %rbx,%rdi
  402552:	44 89 54 24 04       	mov    %r10d,0x4(%rsp)
  402557:	e8 84 e5 ff ff       	callq  400ae0 <fclose@plt>
  40255c:	44 8b 54 24 04       	mov    0x4(%rsp),%r10d
  402561:	e9 d6 fe ff ff       	jmpq   40243c <_Z15mm_read_mtx_crdPcPiS0_S0_PS0_S1_PPdPA4_c+0x5c>
  402566:	8b 45 00             	mov    0x0(%rbp),%eax
  402569:	4c 89 4c 24 08       	mov    %r9,0x8(%rsp)
  40256e:	8d 3c 00             	lea    (%rax,%rax,1),%edi
  402571:	48 63 ff             	movslq %edi,%rdi
  402574:	48 c1 e7 03          	shl    $0x3,%rdi
  402578:	e8 a3 e5 ff ff       	callq  400b20 <malloc@plt>
  40257d:	48 8b 54 24 50       	mov    0x50(%rsp),%rdx
  402582:	48 89 02             	mov    %rax,(%rdx)
  402585:	ff 74 24 58          	pushq  0x58(%rsp)
  402589:	50                   	push   %rax
  40258a:	4c 8b 4c 24 18       	mov    0x18(%rsp),%r9
  40258f:	8b 4d 00             	mov    0x0(%rbp),%ecx
  402592:	41 8b 55 00          	mov    0x0(%r13),%edx
  402596:	48 89 df             	mov    %rbx,%rdi
  402599:	4d 8b 06             	mov    (%r14),%r8
  40259c:	41 8b 34 24          	mov    (%r12),%esi
  4025a0:	e8 5b fc ff ff       	callq  402200 <_Z20mm_read_mtx_crd_dataP8_IO_FILEiiiPiS1_PdPc>
  4025a5:	5a                   	pop    %rdx
  4025a6:	85 c0                	test   %eax,%eax
  4025a8:	59                   	pop    %rcx
  4025a9:	44 8b 54 24 04       	mov    0x4(%rsp),%r10d
  4025ae:	74 92                	je     402542 <_Z15mm_read_mtx_crdPcPiS0_S0_PS0_S1_PPdPA4_c+0x162>
  4025b0:	41 89 c2             	mov    %eax,%r10d
  4025b3:	e9 84 fe ff ff       	jmpq   40243c <_Z15mm_read_mtx_crdPcPiS0_S0_PS0_S1_PPdPA4_c+0x5c>
  4025b8:	4c 89 4c 24 08       	mov    %r9,0x8(%rsp)
  4025bd:	44 89 54 24 04       	mov    %r10d,0x4(%rsp)
  4025c2:	48 63 7d 00          	movslq 0x0(%rbp),%rdi
  4025c6:	eb ac                	jmp    402574 <_Z15mm_read_mtx_crdPcPiS0_S0_PS0_S1_PPdPA4_c+0x194>
  4025c8:	44 89 54 24 04       	mov    %r10d,0x4(%rsp)
  4025cd:	ff 74 24 58          	pushq  0x58(%rsp)
  4025d1:	6a 00                	pushq  $0x0
  4025d3:	eb ba                	jmp    40258f <_Z15mm_read_mtx_crdPcPiS0_S0_PS0_S1_PPdPA4_c+0x1af>
  4025d5:	90                   	nop
  4025d6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4025dd:	00 00 00 

00000000004025e0 <_Z9mm_strdupPKc>:
  4025e0:	55                   	push   %rbp
  4025e1:	53                   	push   %rbx
  4025e2:	48 89 fd             	mov    %rdi,%rbp
  4025e5:	48 83 ec 08          	sub    $0x8,%rsp
  4025e9:	e8 a2 e4 ff ff       	callq  400a90 <strlen@plt>
  4025ee:	8d 78 01             	lea    0x1(%rax),%edi
  4025f1:	48 89 c3             	mov    %rax,%rbx
  4025f4:	48 63 ff             	movslq %edi,%rdi
  4025f7:	e8 24 e5 ff ff       	callq  400b20 <malloc@plt>
  4025fc:	48 83 c4 08          	add    $0x8,%rsp
  402600:	48 8d 53 01          	lea    0x1(%rbx),%rdx
  402604:	48 89 ee             	mov    %rbp,%rsi
  402607:	5b                   	pop    %rbx
  402608:	5d                   	pop    %rbp
  402609:	48 89 c7             	mov    %rax,%rdi
  40260c:	e9 bf e4 ff ff       	jmpq   400ad0 <memcpy@plt>
  402611:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  402616:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40261d:	00 00 00 

0000000000402620 <_Z18mm_typecode_to_strPc>:
  402620:	41 56                	push   %r14
  402622:	41 55                	push   %r13
  402624:	41 54                	push   %r12
  402626:	55                   	push   %rbp
  402627:	53                   	push   %rbx
  402628:	48 89 fb             	mov    %rdi,%rbx
  40262b:	48 81 ec 10 04 00 00 	sub    $0x410,%rsp
  402632:	80 3f 4d             	cmpb   $0x4d,(%rdi)
  402635:	0f 84 7d 01 00 00    	je     4027b8 <_Z18mm_typecode_to_strPc+0x198>
  40263b:	bf 06 00 00 00       	mov    $0x6,%edi
  402640:	e8 db e4 ff ff       	callq  400b20 <malloc@plt>
  402645:	41 b9 72 00 00 00    	mov    $0x72,%r9d
  40264b:	c7 00 65 72 72 6f    	movl   $0x6f727265,(%rax)
  402651:	49 89 c4             	mov    %rax,%r12
  402654:	66 44 89 48 04       	mov    %r9w,0x4(%rax)
  402659:	0f b6 43 01          	movzbl 0x1(%rbx),%eax
  40265d:	3c 43                	cmp    $0x43,%al
  40265f:	0f 84 81 01 00 00    	je     4027e6 <_Z18mm_typecode_to_strPc+0x1c6>
  402665:	3c 41                	cmp    $0x41,%al
  402667:	0f 85 33 01 00 00    	jne    4027a0 <_Z18mm_typecode_to_strPc+0x180>
  40266d:	bf 06 00 00 00       	mov    $0x6,%edi
  402672:	e8 a9 e4 ff ff       	callq  400b20 <malloc@plt>
  402677:	bf 79 00 00 00       	mov    $0x79,%edi
  40267c:	c7 00 61 72 72 61    	movl   $0x61727261,(%rax)
  402682:	48 89 c5             	mov    %rax,%rbp
  402685:	66 89 78 04          	mov    %di,0x4(%rax)
  402689:	0f b6 43 02          	movzbl 0x2(%rbx),%eax
  40268d:	3c 52                	cmp    $0x52,%al
  40268f:	0f 84 ab 01 00 00    	je     402840 <_Z18mm_typecode_to_strPc+0x220>
  402695:	3c 43                	cmp    $0x43,%al
  402697:	0f 84 7b 01 00 00    	je     402818 <_Z18mm_typecode_to_strPc+0x1f8>
  40269d:	3c 50                	cmp    $0x50,%al
  40269f:	0f 84 bb 01 00 00    	je     402860 <_Z18mm_typecode_to_strPc+0x240>
  4026a5:	3c 49                	cmp    $0x49,%al
  4026a7:	0f 85 f3 00 00 00    	jne    4027a0 <_Z18mm_typecode_to_strPc+0x180>
  4026ad:	bf 08 00 00 00       	mov    $0x8,%edi
  4026b2:	e8 69 e4 ff ff       	callq  400b20 <malloc@plt>
  4026b7:	48 b9 69 6e 74 65 67 	movabs $0x72656765746e69,%rcx
  4026be:	65 72 00 
  4026c1:	49 89 c5             	mov    %rax,%r13
  4026c4:	48 89 08             	mov    %rcx,(%rax)
  4026c7:	0f b6 43 03          	movzbl 0x3(%rbx),%eax
  4026cb:	3c 47                	cmp    $0x47,%al
  4026cd:	0f 84 ad 01 00 00    	je     402880 <_Z18mm_typecode_to_strPc+0x260>
  4026d3:	3c 53                	cmp    $0x53,%al
  4026d5:	0f 84 c5 01 00 00    	je     4028a0 <_Z18mm_typecode_to_strPc+0x280>
  4026db:	3c 48                	cmp    $0x48,%al
  4026dd:	0f 84 ed 01 00 00    	je     4028d0 <_Z18mm_typecode_to_strPc+0x2b0>
  4026e3:	3c 4b                	cmp    $0x4b,%al
  4026e5:	0f 85 b5 00 00 00    	jne    4027a0 <_Z18mm_typecode_to_strPc+0x180>
  4026eb:	bf 0f 00 00 00       	mov    $0xf,%edi
  4026f0:	e8 2b e4 ff ff       	callq  400b20 <malloc@plt>
  4026f5:	48 be 73 6b 65 77 2d 	movabs $0x6d79732d77656b73,%rsi
  4026fc:	73 79 6d 
  4026ff:	ba 69 63 00 00       	mov    $0x6369,%edx
  402704:	c7 40 08 6d 65 74 72 	movl   $0x7274656d,0x8(%rax)
  40270b:	48 89 30             	mov    %rsi,(%rax)
  40270e:	66 89 50 0c          	mov    %dx,0xc(%rax)
  402712:	49 89 c1             	mov    %rax,%r9
  402715:	c6 40 0e 00          	movb   $0x0,0xe(%rax)
  402719:	4d 89 e8             	mov    %r13,%r8
  40271c:	48 89 e9             	mov    %rbp,%rcx
  40271f:	4c 89 e2             	mov    %r12,%rdx
  402722:	be 13 2f 40 00       	mov    $0x402f13,%esi
  402727:	48 89 e7             	mov    %rsp,%rdi
  40272a:	31 c0                	xor    %eax,%eax
  40272c:	49 89 e6             	mov    %rsp,%r14
  40272f:	48 89 e3             	mov    %rsp,%rbx
  402732:	e8 49 e3 ff ff       	callq  400a80 <sprintf@plt>
  402737:	8b 13                	mov    (%rbx),%edx
  402739:	48 83 c3 04          	add    $0x4,%rbx
  40273d:	8d 82 ff fe fe fe    	lea    -0x1010101(%rdx),%eax
  402743:	f7 d2                	not    %edx
  402745:	21 d0                	and    %edx,%eax
  402747:	25 80 80 80 80       	and    $0x80808080,%eax
  40274c:	74 e9                	je     402737 <_Z18mm_typecode_to_strPc+0x117>
  40274e:	89 c2                	mov    %eax,%edx
  402750:	c1 ea 10             	shr    $0x10,%edx
  402753:	a9 80 80 00 00       	test   $0x8080,%eax
  402758:	0f 44 c2             	cmove  %edx,%eax
  40275b:	48 8d 53 02          	lea    0x2(%rbx),%rdx
  40275f:	89 c1                	mov    %eax,%ecx
  402761:	48 0f 44 da          	cmove  %rdx,%rbx
  402765:	00 c1                	add    %al,%cl
  402767:	48 83 db 03          	sbb    $0x3,%rbx
  40276b:	4c 29 f3             	sub    %r14,%rbx
  40276e:	8d 7b 01             	lea    0x1(%rbx),%edi
  402771:	48 63 ff             	movslq %edi,%rdi
  402774:	e8 a7 e3 ff ff       	callq  400b20 <malloc@plt>
  402779:	48 8d 53 01          	lea    0x1(%rbx),%rdx
  40277d:	4c 89 f6             	mov    %r14,%rsi
  402780:	48 89 c7             	mov    %rax,%rdi
  402783:	e8 48 e3 ff ff       	callq  400ad0 <memcpy@plt>
  402788:	48 81 c4 10 04 00 00 	add    $0x410,%rsp
  40278f:	5b                   	pop    %rbx
  402790:	5d                   	pop    %rbp
  402791:	41 5c                	pop    %r12
  402793:	41 5d                	pop    %r13
  402795:	41 5e                	pop    %r14
  402797:	c3                   	retq   
  402798:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  40279f:	00 
  4027a0:	48 81 c4 10 04 00 00 	add    $0x410,%rsp
  4027a7:	31 c0                	xor    %eax,%eax
  4027a9:	5b                   	pop    %rbx
  4027aa:	5d                   	pop    %rbp
  4027ab:	41 5c                	pop    %r12
  4027ad:	41 5d                	pop    %r13
  4027af:	41 5e                	pop    %r14
  4027b1:	c3                   	retq   
  4027b2:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  4027b8:	bf 07 00 00 00       	mov    $0x7,%edi
  4027bd:	e8 5e e3 ff ff       	callq  400b20 <malloc@plt>
  4027c2:	41 ba 69 78 00 00    	mov    $0x7869,%r10d
  4027c8:	c7 00 6d 61 74 72    	movl   $0x7274616d,(%rax)
  4027ce:	c6 40 06 00          	movb   $0x0,0x6(%rax)
  4027d2:	66 44 89 50 04       	mov    %r10w,0x4(%rax)
  4027d7:	49 89 c4             	mov    %rax,%r12
  4027da:	0f b6 43 01          	movzbl 0x1(%rbx),%eax
  4027de:	3c 43                	cmp    $0x43,%al
  4027e0:	0f 85 7f fe ff ff    	jne    402665 <_Z18mm_typecode_to_strPc+0x45>
  4027e6:	bf 0b 00 00 00       	mov    $0xb,%edi
  4027eb:	e8 30 e3 ff ff       	callq  400b20 <malloc@plt>
  4027f0:	48 be 63 6f 6f 72 64 	movabs $0x616e6964726f6f63,%rsi
  4027f7:	69 6e 61 
  4027fa:	41 b8 74 65 00 00    	mov    $0x6574,%r8d
  402800:	c6 40 0a 00          	movb   $0x0,0xa(%rax)
  402804:	48 89 30             	mov    %rsi,(%rax)
  402807:	66 44 89 40 08       	mov    %r8w,0x8(%rax)
  40280c:	48 89 c5             	mov    %rax,%rbp
  40280f:	e9 75 fe ff ff       	jmpq   402689 <_Z18mm_typecode_to_strPc+0x69>
  402814:	0f 1f 40 00          	nopl   0x0(%rax)
  402818:	bf 08 00 00 00       	mov    $0x8,%edi
  40281d:	e8 fe e2 ff ff       	callq  400b20 <malloc@plt>
  402822:	48 be 63 6f 6d 70 6c 	movabs $0x78656c706d6f63,%rsi
  402829:	65 78 00 
  40282c:	49 89 c5             	mov    %rax,%r13
  40282f:	48 89 30             	mov    %rsi,(%rax)
  402832:	e9 90 fe ff ff       	jmpq   4026c7 <_Z18mm_typecode_to_strPc+0xa7>
  402837:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  40283e:	00 00 
  402840:	bf 05 00 00 00       	mov    $0x5,%edi
  402845:	e8 d6 e2 ff ff       	callq  400b20 <malloc@plt>
  40284a:	c7 00 72 65 61 6c    	movl   $0x6c616572,(%rax)
  402850:	c6 40 04 00          	movb   $0x0,0x4(%rax)
  402854:	49 89 c5             	mov    %rax,%r13
  402857:	e9 6b fe ff ff       	jmpq   4026c7 <_Z18mm_typecode_to_strPc+0xa7>
  40285c:	0f 1f 40 00          	nopl   0x0(%rax)
  402860:	bf 08 00 00 00       	mov    $0x8,%edi
  402865:	e8 b6 e2 ff ff       	callq  400b20 <malloc@plt>
  40286a:	48 b9 70 61 74 74 65 	movabs $0x6e726574746170,%rcx
  402871:	72 6e 00 
  402874:	49 89 c5             	mov    %rax,%r13
  402877:	48 89 08             	mov    %rcx,(%rax)
  40287a:	e9 48 fe ff ff       	jmpq   4026c7 <_Z18mm_typecode_to_strPc+0xa7>
  40287f:	90                   	nop
  402880:	bf 08 00 00 00       	mov    $0x8,%edi
  402885:	e8 96 e2 ff ff       	callq  400b20 <malloc@plt>
  40288a:	48 be 67 65 6e 65 72 	movabs $0x6c6172656e6567,%rsi
  402891:	61 6c 00 
  402894:	49 89 c1             	mov    %rax,%r9
  402897:	48 89 30             	mov    %rsi,(%rax)
  40289a:	e9 7a fe ff ff       	jmpq   402719 <_Z18mm_typecode_to_strPc+0xf9>
  40289f:	90                   	nop
  4028a0:	bf 0a 00 00 00       	mov    $0xa,%edi
  4028a5:	e8 76 e2 ff ff       	callq  400b20 <malloc@plt>
  4028aa:	48 b9 73 79 6d 6d 65 	movabs $0x697274656d6d7973,%rcx
  4028b1:	74 72 69 
  4028b4:	be 63 00 00 00       	mov    $0x63,%esi
  4028b9:	49 89 c1             	mov    %rax,%r9
  4028bc:	48 89 08             	mov    %rcx,(%rax)
  4028bf:	66 89 70 08          	mov    %si,0x8(%rax)
  4028c3:	e9 51 fe ff ff       	jmpq   402719 <_Z18mm_typecode_to_strPc+0xf9>
  4028c8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  4028cf:	00 
  4028d0:	bf 0a 00 00 00       	mov    $0xa,%edi
  4028d5:	e8 46 e2 ff ff       	callq  400b20 <malloc@plt>
  4028da:	48 b9 68 65 72 6d 69 	movabs $0x616974696d726568,%rcx
  4028e1:	74 69 61 
  4028e4:	49 89 c1             	mov    %rax,%r9
  4028e7:	48 89 08             	mov    %rcx,(%rax)
  4028ea:	b9 6e 00 00 00       	mov    $0x6e,%ecx
  4028ef:	66 89 48 08          	mov    %cx,0x8(%rax)
  4028f3:	e9 21 fe ff ff       	jmpq   402719 <_Z18mm_typecode_to_strPc+0xf9>
  4028f8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  4028ff:	00 

0000000000402900 <_Z26mm_read_unsymmetric_sparsePKcPiS1_S1_PPdPS1_S4_>:
  402900:	41 57                	push   %r15
  402902:	41 56                	push   %r14
  402904:	49 89 f7             	mov    %rsi,%r15
  402907:	41 55                	push   %r13
  402909:	41 54                	push   %r12
  40290b:	be 61 2f 40 00       	mov    $0x402f61,%esi
  402910:	55                   	push   %rbp
  402911:	53                   	push   %rbx
  402912:	48 89 fd             	mov    %rdi,%rbp
  402915:	49 89 d6             	mov    %rdx,%r14
  402918:	49 89 cd             	mov    %rcx,%r13
  40291b:	4d 89 c4             	mov    %r8,%r12
  40291e:	48 83 ec 28          	sub    $0x28,%rsp
  402922:	4c 89 4c 24 08       	mov    %r9,0x8(%rsp)
  402927:	e8 c4 e1 ff ff       	callq  400af0 <fopen@plt>
  40292c:	48 85 c0             	test   %rax,%rax
  40292f:	0f 84 87 01 00 00    	je     402abc <_Z26mm_read_unsymmetric_sparsePKcPiS1_S1_PPdPS1_S4_+0x1bc>
  402935:	48 8d 74 24 10       	lea    0x10(%rsp),%rsi
  40293a:	48 89 c7             	mov    %rax,%rdi
  40293d:	48 89 c3             	mov    %rax,%rbx
  402940:	e8 2b f4 ff ff       	callq  401d70 <_Z14mm_read_bannerP8_IO_FILEPA4_c>
  402945:	85 c0                	test   %eax,%eax
  402947:	0f 85 79 01 00 00    	jne    402ac6 <_Z26mm_read_unsymmetric_sparsePKcPiS1_S1_PPdPS1_S4_+0x1c6>
  40294d:	80 7c 24 12 52       	cmpb   $0x52,0x12(%rsp)
  402952:	0f 85 00 01 00 00    	jne    402a58 <_Z26mm_read_unsymmetric_sparsePKcPiS1_S1_PPdPS1_S4_+0x158>
  402958:	80 7c 24 10 4d       	cmpb   $0x4d,0x10(%rsp)
  40295d:	0f 85 f5 00 00 00    	jne    402a58 <_Z26mm_read_unsymmetric_sparsePKcPiS1_S1_PPdPS1_S4_+0x158>
  402963:	80 7c 24 11 43       	cmpb   $0x43,0x11(%rsp)
  402968:	0f 85 ea 00 00 00    	jne    402a58 <_Z26mm_read_unsymmetric_sparsePKcPiS1_S1_PPdPS1_S4_+0x158>
  40296e:	48 8d 4c 24 1c       	lea    0x1c(%rsp),%rcx
  402973:	48 8d 54 24 18       	lea    0x18(%rsp),%rdx
  402978:	48 8d 74 24 14       	lea    0x14(%rsp),%rsi
  40297d:	48 89 df             	mov    %rbx,%rdi
  402980:	e8 eb f6 ff ff       	callq  402070 <_Z20mm_read_mtx_crd_sizeP8_IO_FILEPiS1_S1_>
  402985:	85 c0                	test   %eax,%eax
  402987:	89 c5                	mov    %eax,%ebp
  402989:	0f 85 0b 01 00 00    	jne    402a9a <_Z26mm_read_unsymmetric_sparsePKcPiS1_S1_PPdPS1_S4_+0x19a>
  40298f:	8b 44 24 14          	mov    0x14(%rsp),%eax
  402993:	48 63 7c 24 1c       	movslq 0x1c(%rsp),%rdi
  402998:	41 89 07             	mov    %eax,(%r15)
  40299b:	8b 44 24 18          	mov    0x18(%rsp),%eax
  40299f:	45 31 ff             	xor    %r15d,%r15d
  4029a2:	41 89 06             	mov    %eax,(%r14)
  4029a5:	41 89 7d 00          	mov    %edi,0x0(%r13)
  4029a9:	48 c1 e7 02          	shl    $0x2,%rdi
  4029ad:	e8 6e e1 ff ff       	callq  400b20 <malloc@plt>
  4029b2:	48 63 7c 24 1c       	movslq 0x1c(%rsp),%rdi
  4029b7:	49 89 c6             	mov    %rax,%r14
  4029ba:	48 c1 e7 02          	shl    $0x2,%rdi
  4029be:	e8 5d e1 ff ff       	callq  400b20 <malloc@plt>
  4029c3:	48 63 7c 24 1c       	movslq 0x1c(%rsp),%rdi
  4029c8:	49 89 c5             	mov    %rax,%r13
  4029cb:	48 c1 e7 03          	shl    $0x3,%rdi
  4029cf:	e8 4c e1 ff ff       	callq  400b20 <malloc@plt>
  4029d4:	48 8b 54 24 60       	mov    0x60(%rsp),%rdx
  4029d9:	48 8b 4c 24 08       	mov    0x8(%rsp),%rcx
  4029de:	49 89 04 24          	mov    %rax,(%r12)
  4029e2:	4d 89 ec             	mov    %r13,%r12
  4029e5:	4c 89 31             	mov    %r14,(%rcx)
  4029e8:	4c 89 2a             	mov    %r13,(%rdx)
  4029eb:	49 89 c5             	mov    %rax,%r13
  4029ee:	8b 54 24 1c          	mov    0x1c(%rsp),%edx
  4029f2:	85 d2                	test   %edx,%edx
  4029f4:	7e 42                	jle    402a38 <_Z26mm_read_unsymmetric_sparsePKcPiS1_S1_PPdPS1_S4_+0x138>
  4029f6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4029fd:	00 00 00 
  402a00:	4d 89 e8             	mov    %r13,%r8
  402a03:	4c 89 e1             	mov    %r12,%rcx
  402a06:	4c 89 f2             	mov    %r14,%rdx
  402a09:	31 c0                	xor    %eax,%eax
  402a0b:	be a5 2f 40 00       	mov    $0x402fa5,%esi
  402a10:	48 89 df             	mov    %rbx,%rdi
  402a13:	e8 98 e1 ff ff       	callq  400bb0 <fscanf@plt>
  402a18:	41 83 c7 01          	add    $0x1,%r15d
  402a1c:	41 83 2e 01          	subl   $0x1,(%r14)
  402a20:	41 83 2c 24 01       	subl   $0x1,(%r12)
  402a25:	49 83 c6 04          	add    $0x4,%r14
  402a29:	49 83 c4 04          	add    $0x4,%r12
  402a2d:	49 83 c5 08          	add    $0x8,%r13
  402a31:	44 39 7c 24 1c       	cmp    %r15d,0x1c(%rsp)
  402a36:	7f c8                	jg     402a00 <_Z26mm_read_unsymmetric_sparsePKcPiS1_S1_PPdPS1_S4_+0x100>
  402a38:	48 89 df             	mov    %rbx,%rdi
  402a3b:	e8 a0 e0 ff ff       	callq  400ae0 <fclose@plt>
  402a40:	48 83 c4 28          	add    $0x28,%rsp
  402a44:	89 e8                	mov    %ebp,%eax
  402a46:	5b                   	pop    %rbx
  402a47:	5d                   	pop    %rbp
  402a48:	41 5c                	pop    %r12
  402a4a:	41 5d                	pop    %r13
  402a4c:	41 5e                	pop    %r14
  402a4e:	41 5f                	pop    %r15
  402a50:	c3                   	retq   
  402a51:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  402a58:	48 8b 0d 91 16 20 00 	mov    0x201691(%rip),%rcx        # 6040f0 <stderr@@GLIBC_2.2.5>
  402a5f:	ba 29 00 00 00       	mov    $0x29,%edx
  402a64:	be 01 00 00 00       	mov    $0x1,%esi
  402a69:	bf 60 30 40 00       	mov    $0x403060,%edi
  402a6e:	bd ff ff ff ff       	mov    $0xffffffff,%ebp
  402a73:	e8 48 e1 ff ff       	callq  400bc0 <fwrite@plt>
  402a78:	48 8d 7c 24 10       	lea    0x10(%rsp),%rdi
  402a7d:	e8 9e fb ff ff       	callq  402620 <_Z18mm_typecode_to_strPc>
  402a82:	48 8b 3d 67 16 20 00 	mov    0x201667(%rip),%rdi        # 6040f0 <stderr@@GLIBC_2.2.5>
  402a89:	48 89 c2             	mov    %rax,%rdx
  402a8c:	be c5 2f 40 00       	mov    $0x402fc5,%esi
  402a91:	31 c0                	xor    %eax,%eax
  402a93:	e8 a8 e0 ff ff       	callq  400b40 <fprintf@plt>
  402a98:	eb a6                	jmp    402a40 <_Z26mm_read_unsymmetric_sparsePKcPiS1_S1_PPdPS1_S4_+0x140>
  402a9a:	48 8b 0d 4f 16 20 00 	mov    0x20164f(%rip),%rcx        # 6040f0 <stderr@@GLIBC_2.2.5>
  402aa1:	ba 38 00 00 00       	mov    $0x38,%edx
  402aa6:	be 01 00 00 00       	mov    $0x1,%esi
  402aab:	bf 90 30 40 00       	mov    $0x403090,%edi
  402ab0:	bd ff ff ff ff       	mov    $0xffffffff,%ebp
  402ab5:	e8 06 e1 ff ff       	callq  400bc0 <fwrite@plt>
  402aba:	eb 84                	jmp    402a40 <_Z26mm_read_unsymmetric_sparsePKcPiS1_S1_PPdPS1_S4_+0x140>
  402abc:	bd ff ff ff ff       	mov    $0xffffffff,%ebp
  402ac1:	e9 7a ff ff ff       	jmpq   402a40 <_Z26mm_read_unsymmetric_sparsePKcPiS1_S1_PPdPS1_S4_+0x140>
  402ac6:	bf 20 30 40 00       	mov    $0x403020,%edi
  402acb:	31 c0                	xor    %eax,%eax
  402acd:	e8 9e df ff ff       	callq  400a70 <printf@plt>
  402ad2:	48 89 ee             	mov    %rbp,%rsi
  402ad5:	bf b6 2f 40 00       	mov    $0x402fb6,%edi
  402ada:	31 c0                	xor    %eax,%eax
  402adc:	e8 8f df ff ff       	callq  400a70 <printf@plt>
  402ae1:	bd ff ff ff ff       	mov    $0xffffffff,%ebp
  402ae6:	e9 55 ff ff ff       	jmpq   402a40 <_Z26mm_read_unsymmetric_sparsePKcPiS1_S1_PPdPS1_S4_+0x140>
  402aeb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000402af0 <_Z15mm_write_bannerP8_IO_FILEPc>:
  402af0:	55                   	push   %rbp
  402af1:	53                   	push   %rbx
  402af2:	48 89 fd             	mov    %rdi,%rbp
  402af5:	48 89 f7             	mov    %rsi,%rdi
  402af8:	48 83 ec 08          	sub    $0x8,%rsp
  402afc:	e8 1f fb ff ff       	callq  402620 <_Z18mm_typecode_to_strPc>
  402b01:	ba 1f 2f 40 00       	mov    $0x402f1f,%edx
  402b06:	48 89 c3             	mov    %rax,%rbx
  402b09:	48 89 c1             	mov    %rax,%rcx
  402b0c:	48 89 ef             	mov    %rbp,%rdi
  402b0f:	be df 2f 40 00       	mov    $0x402fdf,%esi
  402b14:	31 c0                	xor    %eax,%eax
  402b16:	e8 25 e0 ff ff       	callq  400b40 <fprintf@plt>
  402b1b:	48 89 df             	mov    %rbx,%rdi
  402b1e:	89 c5                	mov    %eax,%ebp
  402b20:	e8 db df ff ff       	callq  400b00 <free@plt>
  402b25:	83 fd 02             	cmp    $0x2,%ebp
  402b28:	ba 00 00 00 00       	mov    $0x0,%edx
  402b2d:	b8 11 00 00 00       	mov    $0x11,%eax
  402b32:	0f 44 c2             	cmove  %edx,%eax
  402b35:	48 83 c4 08          	add    $0x8,%rsp
  402b39:	5b                   	pop    %rbx
  402b3a:	5d                   	pop    %rbp
  402b3b:	c3                   	retq   
  402b3c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000402b40 <_Z16mm_write_mtx_crdPciiiPiS0_PdS_>:
  402b40:	41 57                	push   %r15
  402b42:	41 56                	push   %r14
  402b44:	48 89 f8             	mov    %rdi,%rax
  402b47:	41 55                	push   %r13
  402b49:	41 54                	push   %r12
  402b4b:	41 89 ce             	mov    %ecx,%r14d
  402b4e:	55                   	push   %rbp
  402b4f:	53                   	push   %rbx
  402b50:	bf e6 2f 40 00       	mov    $0x402fe6,%edi
  402b55:	b9 07 00 00 00       	mov    $0x7,%ecx
  402b5a:	41 89 d7             	mov    %edx,%r15d
  402b5d:	4c 89 c5             	mov    %r8,%rbp
  402b60:	48 83 ec 18          	sub    $0x18,%rsp
  402b64:	4d 89 cc             	mov    %r9,%r12
  402b67:	48 8b 1d 72 15 20 00 	mov    0x201572(%rip),%rbx        # 6040e0 <stdout@@GLIBC_2.2.5>
  402b6e:	89 74 24 0c          	mov    %esi,0xc(%rsp)
  402b72:	48 89 c6             	mov    %rax,%rsi
  402b75:	4c 8b 6c 24 50       	mov    0x50(%rsp),%r13
  402b7a:	f3 a6                	repz cmpsb %es:(%rdi),%ds:(%rsi)
  402b7c:	0f 85 9e 00 00 00    	jne    402c20 <_Z16mm_write_mtx_crdPciiiPiS0_PdS_+0xe0>
  402b82:	ba 1f 2f 40 00       	mov    $0x402f1f,%edx
  402b87:	be ef 2f 40 00       	mov    $0x402fef,%esi
  402b8c:	48 89 df             	mov    %rbx,%rdi
  402b8f:	31 c0                	xor    %eax,%eax
  402b91:	e8 aa df ff ff       	callq  400b40 <fprintf@plt>
  402b96:	48 8b 7c 24 58       	mov    0x58(%rsp),%rdi
  402b9b:	e8 80 fa ff ff       	callq  402620 <_Z18mm_typecode_to_strPc>
  402ba0:	be e2 2f 40 00       	mov    $0x402fe2,%esi
  402ba5:	48 89 c2             	mov    %rax,%rdx
  402ba8:	48 89 df             	mov    %rbx,%rdi
  402bab:	31 c0                	xor    %eax,%eax
  402bad:	e8 8e df ff ff       	callq  400b40 <fprintf@plt>
  402bb2:	8b 54 24 0c          	mov    0xc(%rsp),%edx
  402bb6:	31 c0                	xor    %eax,%eax
  402bb8:	45 89 f0             	mov    %r14d,%r8d
  402bbb:	44 89 f9             	mov    %r15d,%ecx
  402bbe:	be 84 2f 40 00       	mov    $0x402f84,%esi
  402bc3:	48 89 df             	mov    %rbx,%rdi
  402bc6:	e8 75 df ff ff       	callq  400b40 <fprintf@plt>
  402bcb:	48 8b 44 24 58       	mov    0x58(%rsp),%rax
  402bd0:	0f b6 40 02          	movzbl 0x2(%rax),%eax
  402bd4:	3c 50                	cmp    $0x50,%al
  402bd6:	74 68                	je     402c40 <_Z16mm_write_mtx_crdPciiiPiS0_PdS_+0x100>
  402bd8:	3c 52                	cmp    $0x52,%al
  402bda:	0f 84 a8 00 00 00    	je     402c88 <_Z16mm_write_mtx_crdPciiiPiS0_PdS_+0x148>
  402be0:	3c 43                	cmp    $0x43,%al
  402be2:	0f 84 f8 00 00 00    	je     402ce0 <_Z16mm_write_mtx_crdPciiiPiS0_PdS_+0x1a0>
  402be8:	48 3b 1d f1 14 20 00 	cmp    0x2014f1(%rip),%rbx        # 6040e0 <stdout@@GLIBC_2.2.5>
  402bef:	ba 0f 00 00 00       	mov    $0xf,%edx
  402bf4:	74 10                	je     402c06 <_Z16mm_write_mtx_crdPciiiPiS0_PdS_+0xc6>
  402bf6:	48 89 df             	mov    %rbx,%rdi
  402bf9:	89 54 24 0c          	mov    %edx,0xc(%rsp)
  402bfd:	e8 de de ff ff       	callq  400ae0 <fclose@plt>
  402c02:	8b 54 24 0c          	mov    0xc(%rsp),%edx
  402c06:	48 83 c4 18          	add    $0x18,%rsp
  402c0a:	89 d0                	mov    %edx,%eax
  402c0c:	5b                   	pop    %rbx
  402c0d:	5d                   	pop    %rbp
  402c0e:	41 5c                	pop    %r12
  402c10:	41 5d                	pop    %r13
  402c12:	41 5e                	pop    %r14
  402c14:	41 5f                	pop    %r15
  402c16:	c3                   	retq   
  402c17:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  402c1e:	00 00 
  402c20:	be ed 2f 40 00       	mov    $0x402fed,%esi
  402c25:	48 89 c7             	mov    %rax,%rdi
  402c28:	e8 c3 de ff ff       	callq  400af0 <fopen@plt>
  402c2d:	48 85 c0             	test   %rax,%rax
  402c30:	48 89 c3             	mov    %rax,%rbx
  402c33:	ba 11 00 00 00       	mov    $0x11,%edx
  402c38:	0f 85 44 ff ff ff    	jne    402b82 <_Z16mm_write_mtx_crdPciiiPiS0_PdS_+0x42>
  402c3e:	eb c6                	jmp    402c06 <_Z16mm_write_mtx_crdPciiiPiS0_PdS_+0xc6>
  402c40:	45 85 f6             	test   %r14d,%r14d
  402c43:	7e 2c                	jle    402c71 <_Z16mm_write_mtx_crdPciiiPiS0_PdS_+0x131>
  402c45:	41 83 ee 01          	sub    $0x1,%r14d
  402c49:	45 31 ed             	xor    %r13d,%r13d
  402c4c:	49 83 c6 01          	add    $0x1,%r14
  402c50:	43 8b 0c ac          	mov    (%r12,%r13,4),%ecx
  402c54:	42 8b 54 ad 00       	mov    0x0(%rbp,%r13,4),%edx
  402c59:	31 c0                	xor    %eax,%eax
  402c5b:	be 87 2f 40 00       	mov    $0x402f87,%esi
  402c60:	48 89 df             	mov    %rbx,%rdi
  402c63:	49 83 c5 01          	add    $0x1,%r13
  402c67:	e8 d4 de ff ff       	callq  400b40 <fprintf@plt>
  402c6c:	4d 39 ee             	cmp    %r13,%r14
  402c6f:	75 df                	jne    402c50 <_Z16mm_write_mtx_crdPciiiPiS0_PdS_+0x110>
  402c71:	31 d2                	xor    %edx,%edx
  402c73:	48 3b 1d 66 14 20 00 	cmp    0x201466(%rip),%rbx        # 6040e0 <stdout@@GLIBC_2.2.5>
  402c7a:	0f 85 76 ff ff ff    	jne    402bf6 <_Z16mm_write_mtx_crdPciiiPiS0_PdS_+0xb6>
  402c80:	eb 84                	jmp    402c06 <_Z16mm_write_mtx_crdPciiiPiS0_PdS_+0xc6>
  402c82:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  402c88:	45 85 f6             	test   %r14d,%r14d
  402c8b:	7e e4                	jle    402c71 <_Z16mm_write_mtx_crdPciiiPiS0_PdS_+0x131>
  402c8d:	45 8d 7e ff          	lea    -0x1(%r14),%r15d
  402c91:	45 31 f6             	xor    %r14d,%r14d
  402c94:	49 83 c7 01          	add    $0x1,%r15
  402c98:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  402c9f:	00 
  402ca0:	43 8b 0c b4          	mov    (%r12,%r14,4),%ecx
  402ca4:	42 8b 54 b5 00       	mov    0x0(%rbp,%r14,4),%edx
  402ca9:	be f3 2f 40 00       	mov    $0x402ff3,%esi
  402cae:	f2 43 0f 10 44 f5 00 	movsd  0x0(%r13,%r14,8),%xmm0
  402cb5:	48 89 df             	mov    %rbx,%rdi
  402cb8:	b8 01 00 00 00       	mov    $0x1,%eax
  402cbd:	49 83 c6 01          	add    $0x1,%r14
  402cc1:	e8 7a de ff ff       	callq  400b40 <fprintf@plt>
  402cc6:	4d 39 f7             	cmp    %r14,%r15
  402cc9:	75 d5                	jne    402ca0 <_Z16mm_write_mtx_crdPciiiPiS0_PdS_+0x160>
  402ccb:	31 d2                	xor    %edx,%edx
  402ccd:	48 3b 1d 0c 14 20 00 	cmp    0x20140c(%rip),%rbx        # 6040e0 <stdout@@GLIBC_2.2.5>
  402cd4:	0f 85 1c ff ff ff    	jne    402bf6 <_Z16mm_write_mtx_crdPciiiPiS0_PdS_+0xb6>
  402cda:	e9 27 ff ff ff       	jmpq   402c06 <_Z16mm_write_mtx_crdPciiiPiS0_PdS_+0xc6>
  402cdf:	90                   	nop
  402ce0:	45 85 f6             	test   %r14d,%r14d
  402ce3:	7e 8c                	jle    402c71 <_Z16mm_write_mtx_crdPciiiPiS0_PdS_+0x131>
  402ce5:	41 8d 46 ff          	lea    -0x1(%r14),%eax
  402ce9:	45 31 f6             	xor    %r14d,%r14d
  402cec:	4c 8d 3c 85 04 00 00 	lea    0x4(,%rax,4),%r15
  402cf3:	00 
  402cf4:	0f 1f 40 00          	nopl   0x0(%rax)
  402cf8:	43 8b 0c 34          	mov    (%r12,%r14,1),%ecx
  402cfc:	42 8b 54 35 00       	mov    0x0(%rbp,%r14,1),%edx
  402d01:	be 02 30 40 00       	mov    $0x403002,%esi
  402d06:	f2 43 0f 10 4c b5 08 	movsd  0x8(%r13,%r14,4),%xmm1
  402d0d:	48 89 df             	mov    %rbx,%rdi
  402d10:	f2 43 0f 10 44 b5 00 	movsd  0x0(%r13,%r14,4),%xmm0
  402d17:	b8 02 00 00 00       	mov    $0x2,%eax
  402d1c:	49 83 c6 04          	add    $0x4,%r14
  402d20:	e8 1b de ff ff       	callq  400b40 <fprintf@plt>
  402d25:	4d 39 f7             	cmp    %r14,%r15
  402d28:	75 ce                	jne    402cf8 <_Z16mm_write_mtx_crdPciiiPiS0_PdS_+0x1b8>
  402d2a:	31 d2                	xor    %edx,%edx
  402d2c:	48 3b 1d ad 13 20 00 	cmp    0x2013ad(%rip),%rbx        # 6040e0 <stdout@@GLIBC_2.2.5>
  402d33:	0f 85 bd fe ff ff    	jne    402bf6 <_Z16mm_write_mtx_crdPciiiPiS0_PdS_+0xb6>
  402d39:	e9 c8 fe ff ff       	jmpq   402c06 <_Z16mm_write_mtx_crdPciiiPiS0_PdS_+0xc6>
  402d3e:	66 90                	xchg   %ax,%ax

0000000000402d40 <__libc_csu_init>:
  402d40:	41 57                	push   %r15
  402d42:	41 89 ff             	mov    %edi,%r15d
  402d45:	41 56                	push   %r14
  402d47:	49 89 f6             	mov    %rsi,%r14
  402d4a:	41 55                	push   %r13
  402d4c:	49 89 d5             	mov    %rdx,%r13
  402d4f:	41 54                	push   %r12
  402d51:	4c 8d 25 68 10 20 00 	lea    0x201068(%rip),%r12        # 603dc0 <__frame_dummy_init_array_entry>
  402d58:	55                   	push   %rbp
  402d59:	48 8d 2d 68 10 20 00 	lea    0x201068(%rip),%rbp        # 603dc8 <__init_array_end>
  402d60:	53                   	push   %rbx
  402d61:	4c 29 e5             	sub    %r12,%rbp
  402d64:	31 db                	xor    %ebx,%ebx
  402d66:	48 c1 fd 03          	sar    $0x3,%rbp
  402d6a:	48 83 ec 08          	sub    $0x8,%rsp
  402d6e:	e8 c5 dc ff ff       	callq  400a38 <_init>
  402d73:	48 85 ed             	test   %rbp,%rbp
  402d76:	74 1e                	je     402d96 <__libc_csu_init+0x56>
  402d78:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  402d7f:	00 
  402d80:	4c 89 ea             	mov    %r13,%rdx
  402d83:	4c 89 f6             	mov    %r14,%rsi
  402d86:	44 89 ff             	mov    %r15d,%edi
  402d89:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  402d8d:	48 83 c3 01          	add    $0x1,%rbx
  402d91:	48 39 eb             	cmp    %rbp,%rbx
  402d94:	75 ea                	jne    402d80 <__libc_csu_init+0x40>
  402d96:	48 83 c4 08          	add    $0x8,%rsp
  402d9a:	5b                   	pop    %rbx
  402d9b:	5d                   	pop    %rbp
  402d9c:	41 5c                	pop    %r12
  402d9e:	41 5d                	pop    %r13
  402da0:	41 5e                	pop    %r14
  402da2:	41 5f                	pop    %r15
  402da4:	c3                   	retq   
  402da5:	90                   	nop
  402da6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  402dad:	00 00 00 

0000000000402db0 <__libc_csu_fini>:
  402db0:	f3 c3                	repz retq 
  402db2:	66 90                	xchg   %ax,%ax

Disassembly of section .fini:

0000000000402db4 <_fini>:
  402db4:	48 83 ec 08          	sub    $0x8,%rsp
  402db8:	48 83 c4 08          	add    $0x8,%rsp
  402dbc:	c3                   	retq   
