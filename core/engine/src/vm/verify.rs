//! Structural check of register bytecode at the bytecompiler boundary.
//!
//! Decoding uses `Bytecode::next_instruction`. `ByteCompiler::finish` calls
//! `verify` only when debug assertions are enabled.

#![cfg_attr(not(debug_assertions), allow(dead_code))]

use std::collections::HashSet;

use super::{
    CodeBlock, Constant,
    opcode::{Address, Instruction, Opcode, RegisterOperand},
};

/// What an opcode requires a constant-pool slot to be.
#[derive(Clone, Copy)]
enum Kind {
    String,
    Function,
    Scope,
    StringOrBigInt,
}

impl Kind {
    fn matches(self, constant: &Constant) -> bool {
        matches!(
            (self, constant),
            (Self::String | Self::StringOrBigInt, Constant::String(_))
                | (Self::Function, Constant::Function(_))
                | (Self::Scope, Constant::Scope(_))
                | (Self::StringOrBigInt, Constant::BigInt(_))
        )
    }

    fn label(self) -> &'static str {
        match self {
            Self::String => "String",
            Self::Function => "Function",
            Self::Scope => "Scope",
            Self::StringOrBigInt => "String or BigInt",
        }
    }
}

fn constant_name(constant: &Constant) -> &'static str {
    match constant {
        Constant::String(_) => "String",
        Constant::Function(_) => "Function",
        Constant::BigInt(_) => "BigInt",
        Constant::Scope(_) => "Scope",
    }
}

/// Operand width after the opcode byte.
///
/// `RegisterOperand`, `IndexOperand`, `Address`, `u32`, `i32` and `f32` are 4
/// bytes. `i8` is 1, `i16` is 2, `f64` and `u64` are 8. A `ThinVec` is a `u32`
/// length plus that many 4-byte elements. Lengths are little-endian, matching
/// `BytecodeEmitter`.
enum Width {
    Fixed(usize),
    Vec4 { header: usize },
}

fn width(opcode: Opcode) -> Width {
    match opcode {
        Opcode::Pop
        | Opcode::DeleteSuperThrow
        | Opcode::ReThrow
        | Opcode::SuperCallSpread
        | Opcode::SuperCallDerived
        | Opcode::CallSpread
        | Opcode::NewSpread
        | Opcode::CheckReturn
        | Opcode::Return
        | Opcode::AsyncGeneratorClose
        | Opcode::Generator
        | Opcode::AsyncGenerator
        | Opcode::PopEnvironment
        | Opcode::IncrementLoopIteration
        | Opcode::IteratorNext
        | Opcode::CreatePromiseCapability
        | Opcode::PopPrivateEnvironment
        | Opcode::Reserved1
        | Opcode::Reserved2
        | Opcode::Reserved3
        | Opcode::Reserved4
        | Opcode::Reserved5
        | Opcode::Reserved6
        | Opcode::Reserved7
        | Opcode::Reserved8
        | Opcode::Reserved9
        | Opcode::Reserved10
        | Opcode::Reserved11
        | Opcode::Reserved12
        | Opcode::Reserved13
        | Opcode::Reserved14
        | Opcode::Reserved15
        | Opcode::Reserved16
        | Opcode::Reserved17
        | Opcode::Reserved18
        | Opcode::Reserved19
        | Opcode::Reserved20
        | Opcode::Reserved21
        | Opcode::Reserved22
        | Opcode::Reserved23
        | Opcode::Reserved24
        | Opcode::Reserved25
        | Opcode::Reserved26
        | Opcode::Reserved27
        | Opcode::Reserved28
        | Opcode::Reserved29
        | Opcode::Reserved30
        | Opcode::Reserved31
        | Opcode::Reserved32
        | Opcode::Reserved33
        | Opcode::Reserved34
        | Opcode::Reserved35
        | Opcode::Reserved36
        | Opcode::Reserved37
        | Opcode::Reserved38
        | Opcode::Reserved39
        | Opcode::Reserved40
        | Opcode::Reserved41
        | Opcode::Reserved42
        | Opcode::Reserved43
        | Opcode::Reserved44
        | Opcode::Reserved45
        | Opcode::Reserved46
        | Opcode::Reserved47
        | Opcode::Reserved48
        | Opcode::Reserved49
        | Opcode::Reserved50
        | Opcode::Reserved51
        | Opcode::Reserved52
        | Opcode::Reserved53
        | Opcode::Reserved54
        | Opcode::Reserved55
        | Opcode::Reserved56
        | Opcode::Reserved57 => Width::Fixed(0),

        Opcode::StoreZero
        | Opcode::StoreOne
        | Opcode::StoreNan
        | Opcode::StorePositiveInfinity
        | Opcode::StoreNegativeInfinity
        | Opcode::StoreNull
        | Opcode::StoreTrue
        | Opcode::StoreFalse
        | Opcode::StoreUndefined
        | Opcode::StoreEmptyObject
        | Opcode::GetHomeObject
        | Opcode::GetPrototype
        | Opcode::StoreNewArray
        | Opcode::PushElisionToArray
        | Opcode::PushIteratorToArray
        | Opcode::BitNot
        | Opcode::TypeOf
        | Opcode::LogicalNot
        | Opcode::Pos
        | Opcode::Neg
        | Opcode::DefVar
        | Opcode::ThrowMutateImmutable
        | Opcode::GetLocator
        | Opcode::SetNameByLocator
        | Opcode::Jump
        | Opcode::Throw
        | Opcode::Exception
        | Opcode::ThrowNewTypeError
        | Opcode::ThrowNewReferenceError
        | Opcode::GetFunctionObject
        | Opcode::This
        | Opcode::SuperCall
        | Opcode::BindThisValue
        | Opcode::CallEvalSpread
        | Opcode::Call
        | Opcode::New
        | Opcode::SetAccumulator
        | Opcode::SetRegisterFromAccumulator
        | Opcode::PopIntoRegister
        | Opcode::PushFromRegister
        | Opcode::PushScope
        | Opcode::PushObjectEnvironment
        | Opcode::CreateForInIterator
        | Opcode::GetIterator
        | Opcode::GetAsyncIterator
        | Opcode::IteratorUpdateResult
        | Opcode::IteratorDone
        | Opcode::IteratorValue
        | Opcode::IteratorResult
        | Opcode::IteratorToArray
        | Opcode::IteratorStackEmpty
        | Opcode::ValueNotNullOrUndefined
        | Opcode::RestParameterInit
        | Opcode::GeneratorYield
        | Opcode::AsyncGeneratorYield
        | Opcode::Await
        | Opcode::NewTarget
        | Opcode::ImportMeta
        | Opcode::IsObject
        | Opcode::CreateMappedArgumentsObject
        | Opcode::CreateUnmappedArgumentsObject
        | Opcode::DefEvalVar => Width::Fixed(4),

        Opcode::StoreInt8 => Width::Fixed(5),
        Opcode::StoreInt16 => Width::Fixed(6),

        Opcode::StoreInt32
        | Opcode::StoreFloat
        | Opcode::StoreLiteral
        | Opcode::SetHomeObject
        | Opcode::SetPrototype
        | Opcode::PushValueToArray
        | Opcode::LogicalAnd
        | Opcode::LogicalOr
        | Opcode::Coalesce
        | Opcode::Inc
        | Opcode::Dec
        | Opcode::DefInitVar
        | Opcode::PutLexicalValue
        | Opcode::GetArgument
        | Opcode::GetName
        | Opcode::GetNameAndLocator
        | Opcode::GetNameOrUndefined
        | Opcode::SetName
        | Opcode::DeleteName
        | Opcode::GetMethod
        | Opcode::DeletePropertyByName
        | Opcode::DeletePropertyByValue
        | Opcode::ToPropertyKey
        | Opcode::JumpIfTrue
        | Opcode::JumpIfFalse
        | Opcode::JumpIfNotUndefined
        | Opcode::JumpIfNullOrUndefined
        | Opcode::MaybeException
        | Opcode::ThisForObjectEnvironmentName
        | Opcode::GetFunction
        | Opcode::CallEval
        | Opcode::Move
        | Opcode::IteratorPop
        | Opcode::IteratorPush
        | Opcode::IteratorFinishAsyncNext
        | Opcode::CreateIteratorResult
        | Opcode::IteratorReturn => Width::Fixed(8),

        Opcode::StoreDouble
        | Opcode::StoreRegexp
        | Opcode::StoreClassPrototype
        | Opcode::SetClassPrototype
        | Opcode::Add
        | Opcode::Sub
        | Opcode::Div
        | Opcode::Mul
        | Opcode::Mod
        | Opcode::Pow
        | Opcode::ShiftRight
        | Opcode::ShiftLeft
        | Opcode::UnsignedShiftRight
        | Opcode::BitOr
        | Opcode::BitAnd
        | Opcode::BitXor
        | Opcode::In
        | Opcode::InPrivate
        | Opcode::Eq
        | Opcode::StrictEq
        | Opcode::NotEq
        | Opcode::StrictNotEq
        | Opcode::GreaterThan
        | Opcode::GreaterThanOrEq
        | Opcode::LessThan
        | Opcode::LessThanOrEq
        | Opcode::InstanceOf
        | Opcode::GetNameGlobal
        | Opcode::GetLengthProperty
        | Opcode::GetPropertyByName
        | Opcode::SetPropertyByName
        | Opcode::SetFunctionName
        | Opcode::DefineOwnPropertyByName
        | Opcode::DefineClassStaticMethodByName
        | Opcode::DefineClassMethodByName
        | Opcode::DefineOwnPropertyByValue
        | Opcode::DefineClassStaticMethodByValue
        | Opcode::DefineClassMethodByValue
        | Opcode::SetPropertyGetterByName
        | Opcode::DefineClassStaticGetterByName
        | Opcode::DefineClassGetterByName
        | Opcode::SetPropertyGetterByValue
        | Opcode::DefineClassStaticGetterByValue
        | Opcode::DefineClassGetterByValue
        | Opcode::SetPropertySetterByName
        | Opcode::DefineClassStaticSetterByName
        | Opcode::DefineClassSetterByName
        | Opcode::SetPropertySetterByValue
        | Opcode::DefineClassStaticSetterByValue
        | Opcode::DefineClassSetterByValue
        | Opcode::SetPrivateField
        | Opcode::DefinePrivateField
        | Opcode::SetPrivateMethod
        | Opcode::SetPrivateSetter
        | Opcode::SetPrivateGetter
        | Opcode::GetPrivateField
        | Opcode::PushClassFieldPrivate
        | Opcode::PushClassPrivateGetter
        | Opcode::PushClassPrivateSetter
        | Opcode::JumpIfNotLessThan
        | Opcode::JumpIfNotLessThanOrEqual
        | Opcode::JumpIfNotGreaterThan
        | Opcode::JumpIfNotGreaterThanOrEqual
        | Opcode::JumpIfNotEqual
        | Opcode::ImportCall
        | Opcode::Case
        | Opcode::GetPropertyByIndex
        | Opcode::SetPropertyByIndex => Width::Fixed(12),

        Opcode::GetPropertyByNameWithThis
        | Opcode::GetPropertyByValue
        | Opcode::GetPropertyByValuePush
        | Opcode::SetPropertyByNameWithThis
        | Opcode::SetPropertyByValue
        | Opcode::PushClassField
        | Opcode::PushClassPrivateMethod
        | Opcode::TemplateLookup => Width::Fixed(16),

        Opcode::JumpTable | Opcode::ConcatToString | Opcode::PushPrivateEnvironment => {
            Width::Vec4 { header: 4 }
        }
        Opcode::CopyDataProperties => Width::Vec4 { header: 8 },
        Opcode::TemplateCreate => Width::Vec4 { header: 12 },
    }
}

fn read_u32(bytes: &[u8], offset: usize) -> Option<u32> {
    let end = offset.checked_add(4)?;
    let slice = bytes.get(offset..end)?;
    Some(u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]))
}

/// Absolute end offset of the instruction at `pc`, if its operands fit.
fn operand_end(bytes: &[u8], pc: usize, opcode: Opcode) -> Option<usize> {
    let cursor = pc.checked_add(1)?;
    match width(opcode) {
        Width::Fixed(size) => {
            let end = cursor.checked_add(size)?;
            (end <= bytes.len()).then_some(end)
        }
        Width::Vec4 { header } => {
            let len_at = cursor.checked_add(header)?;
            let count = usize::try_from(read_u32(bytes, len_at)?).ok()?;
            let payload = count.checked_mul(4)?;
            let end = len_at.checked_add(4)?.checked_add(payload)?;
            (end <= bytes.len()).then_some(end)
        }
    }
}

struct Decoded {
    pc: u32,
    opcode: Opcode,
    instruction: Instruction,
}

fn decode(code: &CodeBlock, errors: &mut Vec<String>) -> Vec<Decoded> {
    let bytes = &code.bytecode.bytes;
    let mut pc = 0usize;
    let mut decoded = Vec::new();
    while pc < bytes.len() {
        let opcode = Opcode::decode(bytes[pc]);
        let op = opcode.as_str();
        let Ok(pc_u32) = u32::try_from(pc) else {
            errors.push(format!(
                "pc {pc:x}: {op}: bytecode offset does not fit in an address"
            ));
            break;
        };
        let Some(end) = operand_end(bytes, pc, opcode) else {
            errors.push(format!(
                "pc {pc_u32:06x}: {op}: operand extends past bytecode end {:06x}",
                bytes.len()
            ));
            break;
        };
        let (instruction, decoded_end) = code.bytecode.next_instruction(pc);
        if decoded_end != end {
            errors.push(format!(
                "pc {pc_u32:06x}: {op}: decoder advanced to {decoded_end:06x}, operand span ends at {end:06x}"
            ));
            break;
        }
        if decoded_end <= pc {
            errors.push(format!("pc {pc_u32:06x}: {op}: decoder did not advance"));
            break;
        }
        decoded.push(Decoded {
            pc: pc_u32,
            opcode,
            instruction,
        });
        pc = decoded_end;
    }
    decoded
}

fn register(
    errors: &mut Vec<String>,
    pc: u32,
    op: &str,
    register_count: u32,
    reg: RegisterOperand,
) {
    let index = u32::from(reg);
    if index >= register_count {
        errors.push(format!(
            "pc {pc:06x}: {op}: register r{index:02} >= register_count {register_count}"
        ));
    }
}

fn register_u32(errors: &mut Vec<String>, pc: u32, op: &str, register_count: u32, index: u32) {
    if index >= register_count {
        errors.push(format!(
            "pc {pc:06x}: {op}: register r{index:02} >= register_count {register_count}"
        ));
    }
}

fn binding(errors: &mut Vec<String>, pc: u32, op: &str, index: u32, len: usize) {
    if index as usize >= len {
        errors.push(format!(
            "pc {pc:06x}: {op}: binding index {index} >= bindings length {len}"
        ));
    }
}

fn inline_cache(errors: &mut Vec<String>, pc: u32, op: &str, index: u32, len: usize) {
    if index as usize >= len {
        errors.push(format!(
            "pc {pc:06x}: {op}: inline cache index {index} >= ic length {len}"
        ));
    }
}

fn constant(
    errors: &mut Vec<String>,
    pc: u32,
    op: &str,
    code: &CodeBlock,
    index: u32,
    expected: Kind,
) {
    let len = code.constants.len();
    let Some(slot) = code.constants.get(index as usize) else {
        errors.push(format!(
            "pc {pc:06x}: {op}: constant index {index} >= constants length {len}"
        ));
        return;
    };
    if !expected.matches(slot) {
        errors.push(format!(
            "pc {pc:06x}: {op}: constant index {index} is {}, expected {}",
            constant_name(slot),
            expected.label(),
        ));
    }
}

fn constant_at(errors: &mut Vec<String>, at: &str, code: &CodeBlock, index: u32, expected: Kind) {
    let len = code.constants.len();
    let Some(slot) = code.constants.get(index as usize) else {
        errors.push(format!(
            "{at}: constant index {index} >= constants length {len}"
        ));
        return;
    };
    if !expected.matches(slot) {
        errors.push(format!(
            "{at}: constant index {index} is {}, expected {}",
            constant_name(slot),
            expected.label(),
        ));
    }
}

fn jump(errors: &mut Vec<String>, pc: u32, op: &str, starts: &HashSet<u32>, address: Address) {
    let raw = address.as_u32();
    if !starts.contains(&raw) {
        errors.push(format!(
            "pc {pc:06x}: {op}: jump target {raw:06x} is not an instruction start"
        ));
    }
}

fn check_instruction(
    code: &CodeBlock,
    starts: &HashSet<u32>,
    item: &Decoded,
    errors: &mut Vec<String>,
) {
    let pc = item.pc;
    let op = item.opcode.as_str();
    let register_count = code.register_count;
    let bindings_len = code.bindings.len();
    let ic_len = code.ic.len();
    let reg = |errors: &mut Vec<String>, operand: RegisterOperand| {
        register(errors, pc, op, register_count, operand);
    };

    match &item.instruction {
        Instruction::Reserved1
        | Instruction::Reserved2
        | Instruction::Reserved3
        | Instruction::Reserved4
        | Instruction::Reserved5
        | Instruction::Reserved6
        | Instruction::Reserved7
        | Instruction::Reserved8
        | Instruction::Reserved9
        | Instruction::Reserved10
        | Instruction::Reserved11
        | Instruction::Reserved12
        | Instruction::Reserved13
        | Instruction::Reserved14
        | Instruction::Reserved15
        | Instruction::Reserved16
        | Instruction::Reserved17
        | Instruction::Reserved18
        | Instruction::Reserved19
        | Instruction::Reserved20
        | Instruction::Reserved21
        | Instruction::Reserved22
        | Instruction::Reserved23
        | Instruction::Reserved24
        | Instruction::Reserved25
        | Instruction::Reserved26
        | Instruction::Reserved27
        | Instruction::Reserved28
        | Instruction::Reserved29
        | Instruction::Reserved30
        | Instruction::Reserved31
        | Instruction::Reserved32
        | Instruction::Reserved33
        | Instruction::Reserved34
        | Instruction::Reserved35
        | Instruction::Reserved36
        | Instruction::Reserved37
        | Instruction::Reserved38
        | Instruction::Reserved39
        | Instruction::Reserved40
        | Instruction::Reserved41
        | Instruction::Reserved42
        | Instruction::Reserved43
        | Instruction::Reserved44
        | Instruction::Reserved45
        | Instruction::Reserved46
        | Instruction::Reserved47
        | Instruction::Reserved48
        | Instruction::Reserved49
        | Instruction::Reserved50
        | Instruction::Reserved51
        | Instruction::Reserved52
        | Instruction::Reserved53
        | Instruction::Reserved54
        | Instruction::Reserved55
        | Instruction::Reserved56
        | Instruction::Reserved57 => {
            errors.push(format!("pc {pc:06x}: {op}: reserved opcode"));
        }

        Instruction::Pop
        | Instruction::DeleteSuperThrow
        | Instruction::ReThrow
        | Instruction::SuperCallSpread
        | Instruction::SuperCallDerived
        | Instruction::CallSpread
        | Instruction::NewSpread
        | Instruction::CheckReturn
        | Instruction::Return
        | Instruction::AsyncGeneratorClose
        | Instruction::Generator
        | Instruction::AsyncGenerator
        | Instruction::PopEnvironment
        | Instruction::IncrementLoopIteration
        | Instruction::IteratorNext
        | Instruction::CreatePromiseCapability
        | Instruction::PopPrivateEnvironment
        | Instruction::Call { .. }
        | Instruction::New { .. }
        | Instruction::SuperCall { .. } => {}

        Instruction::StoreZero { dst }
        | Instruction::StoreOne { dst }
        | Instruction::StoreNan { dst }
        | Instruction::StorePositiveInfinity { dst }
        | Instruction::StoreNegativeInfinity { dst }
        | Instruction::StoreNull { dst }
        | Instruction::StoreTrue { dst }
        | Instruction::StoreFalse { dst }
        | Instruction::StoreUndefined { dst }
        | Instruction::StoreEmptyObject { dst }
        | Instruction::StoreNewArray { dst }
        | Instruction::Exception { dst }
        | Instruction::This { dst }
        | Instruction::NewTarget { dst }
        | Instruction::ImportMeta { dst }
        | Instruction::CreateMappedArgumentsObject { dst }
        | Instruction::CreateUnmappedArgumentsObject { dst }
        | Instruction::RestParameterInit { dst }
        | Instruction::IteratorDone { dst }
        | Instruction::IteratorValue { dst }
        | Instruction::IteratorResult { dst }
        | Instruction::IteratorToArray { dst }
        | Instruction::IteratorStackEmpty { dst }
        | Instruction::SetRegisterFromAccumulator { dst }
        | Instruction::PopIntoRegister { dst }
        | Instruction::StoreInt8 { dst, .. }
        | Instruction::StoreInt16 { dst, .. }
        | Instruction::StoreInt32 { dst, .. }
        | Instruction::StoreFloat { dst, .. }
        | Instruction::StoreDouble { dst, .. }
        | Instruction::GetArgument { dst, .. } => reg(errors, *dst),

        Instruction::SetAccumulator { src }
        | Instruction::PushFromRegister { src }
        | Instruction::Throw { src }
        | Instruction::SetNameByLocator { src }
        | Instruction::PushObjectEnvironment { src }
        | Instruction::CreateForInIterator { src }
        | Instruction::GetIterator { src }
        | Instruction::GetAsyncIterator { src }
        | Instruction::ValueNotNullOrUndefined { src }
        | Instruction::GeneratorYield { src }
        | Instruction::AsyncGeneratorYield { src }
        | Instruction::Await { src } => reg(errors, *src),

        Instruction::TypeOf { value }
        | Instruction::LogicalNot { value }
        | Instruction::Pos { value }
        | Instruction::Neg { value }
        | Instruction::IsObject { value }
        | Instruction::BindThisValue { value }
        | Instruction::BitNot { value }
        | Instruction::CreateIteratorResult { value, .. } => reg(errors, *value),

        Instruction::PushElisionToArray { array } | Instruction::PushIteratorToArray { array } => {
            reg(errors, *array);
        }
        Instruction::GetHomeObject { function } => reg(errors, *function),
        Instruction::GetPrototype { object } => reg(errors, *object),
        Instruction::IteratorUpdateResult { result } => reg(errors, *result),
        Instruction::GetFunctionObject { function_object } => reg(errors, *function_object),

        Instruction::Inc { src, dst }
        | Instruction::Dec { src, dst }
        | Instruction::Move { src, dst }
        | Instruction::ToPropertyKey { src, dst } => {
            reg(errors, *src);
            reg(errors, *dst);
        }

        Instruction::Add { lhs, rhs, dst }
        | Instruction::Sub { lhs, rhs, dst }
        | Instruction::Div { lhs, rhs, dst }
        | Instruction::Mul { lhs, rhs, dst }
        | Instruction::Mod { lhs, rhs, dst }
        | Instruction::Pow { lhs, rhs, dst }
        | Instruction::ShiftRight { lhs, rhs, dst }
        | Instruction::ShiftLeft { lhs, rhs, dst }
        | Instruction::UnsignedShiftRight { lhs, rhs, dst }
        | Instruction::BitOr { lhs, rhs, dst }
        | Instruction::BitAnd { lhs, rhs, dst }
        | Instruction::BitXor { lhs, rhs, dst }
        | Instruction::In { lhs, rhs, dst }
        | Instruction::Eq { lhs, rhs, dst }
        | Instruction::StrictEq { lhs, rhs, dst }
        | Instruction::NotEq { lhs, rhs, dst }
        | Instruction::StrictNotEq { lhs, rhs, dst }
        | Instruction::GreaterThan { lhs, rhs, dst }
        | Instruction::GreaterThanOrEq { lhs, rhs, dst }
        | Instruction::LessThan { lhs, rhs, dst }
        | Instruction::LessThanOrEq { lhs, rhs, dst }
        | Instruction::InstanceOf { lhs, rhs, dst } => {
            reg(errors, *lhs);
            reg(errors, *rhs);
            reg(errors, *dst);
        }

        Instruction::SetHomeObject { function, home } => {
            reg(errors, *function);
            reg(errors, *home);
        }
        Instruction::SetPrototype { object, prototype } => {
            reg(errors, *object);
            reg(errors, *prototype);
        }
        Instruction::PushValueToArray { value, array } => {
            reg(errors, *value);
            reg(errors, *array);
        }
        Instruction::MaybeException {
            has_exception,
            exception,
        } => {
            reg(errors, *has_exception);
            reg(errors, *exception);
        }
        Instruction::IteratorPop { iterator, next }
        | Instruction::IteratorPush { iterator, next } => {
            reg(errors, *iterator);
            reg(errors, *next);
        }
        Instruction::IteratorFinishAsyncNext { resume_kind, value } => {
            reg(errors, *resume_kind);
            reg(errors, *value);
        }
        Instruction::IteratorReturn { value, called } => {
            reg(errors, *value);
            reg(errors, *called);
        }
        Instruction::DeletePropertyByValue { object, key } => {
            reg(errors, *object);
            reg(errors, *key);
        }

        Instruction::StoreClassPrototype {
            dst,
            class,
            superclass,
        } => {
            reg(errors, *dst);
            reg(errors, *class);
            reg(errors, *superclass);
        }
        Instruction::SetClassPrototype {
            dst,
            prototype,
            class,
        } => {
            reg(errors, *dst);
            reg(errors, *prototype);
            reg(errors, *class);
        }

        Instruction::GetPropertyByValue {
            dst,
            key,
            receiver,
            object,
        }
        | Instruction::GetPropertyByValuePush {
            dst,
            key,
            receiver,
            object,
        } => {
            reg(errors, *dst);
            reg(errors, *key);
            reg(errors, *receiver);
            reg(errors, *object);
        }
        Instruction::SetPropertyByValue {
            value,
            key,
            receiver,
            object,
        } => {
            reg(errors, *value);
            reg(errors, *key);
            reg(errors, *receiver);
            reg(errors, *object);
        }

        Instruction::DefineOwnPropertyByValue { value, key, object }
        | Instruction::DefineClassStaticMethodByValue { value, key, object }
        | Instruction::DefineClassMethodByValue { value, key, object }
        | Instruction::SetPropertyGetterByValue { value, key, object }
        | Instruction::DefineClassStaticGetterByValue { value, key, object }
        | Instruction::DefineClassGetterByValue { value, key, object }
        | Instruction::SetPropertySetterByValue { value, key, object }
        | Instruction::DefineClassStaticSetterByValue { value, key, object }
        | Instruction::DefineClassSetterByValue { value, key, object } => {
            reg(errors, *value);
            reg(errors, *key);
            reg(errors, *object);
        }

        Instruction::JumpIfTrue { address, value }
        | Instruction::JumpIfFalse { address, value }
        | Instruction::JumpIfNotUndefined { address, value }
        | Instruction::JumpIfNullOrUndefined { address, value }
        | Instruction::LogicalAnd { address, value }
        | Instruction::LogicalOr { address, value }
        | Instruction::Coalesce { address, value } => {
            jump(errors, pc, op, starts, *address);
            reg(errors, *value);
        }
        Instruction::JumpIfNotLessThan { address, lhs, rhs }
        | Instruction::JumpIfNotLessThanOrEqual { address, lhs, rhs }
        | Instruction::JumpIfNotGreaterThan { address, lhs, rhs }
        | Instruction::JumpIfNotGreaterThanOrEqual { address, lhs, rhs }
        | Instruction::JumpIfNotEqual { address, lhs, rhs } => {
            jump(errors, pc, op, starts, *address);
            reg(errors, *lhs);
            reg(errors, *rhs);
        }
        Instruction::Jump { address } => jump(errors, pc, op, starts, *address),
        Instruction::Case {
            address,
            value,
            condition,
        } => {
            jump(errors, pc, op, starts, *address);
            reg(errors, *value);
            reg(errors, *condition);
        }
        Instruction::TemplateLookup { address, dst, .. } => {
            jump(errors, pc, op, starts, *address);
            reg(errors, *dst);
        }
        Instruction::JumpTable { index, addresses } => {
            register_u32(errors, pc, op, register_count, *index);
            for address in addresses {
                jump(errors, pc, op, starts, *address);
            }
        }

        Instruction::StoreLiteral { dst, index } => {
            reg(errors, *dst);
            constant(
                errors,
                pc,
                op,
                code,
                u32::from(*index),
                Kind::StringOrBigInt,
            );
        }
        Instruction::GetFunction { dst, index } => {
            reg(errors, *dst);
            constant(errors, pc, op, code, u32::from(*index), Kind::Function);
        }
        Instruction::InPrivate { dst, index, rhs } => {
            reg(errors, *dst);
            reg(errors, *rhs);
            constant(errors, pc, op, code, u32::from(*index), Kind::String);
        }
        Instruction::StoreRegexp {
            dst,
            pattern_index,
            flags_index,
        } => {
            reg(errors, *dst);
            constant(
                errors,
                pc,
                op,
                code,
                u32::from(*pattern_index),
                Kind::String,
            );
            constant(errors, pc, op, code, u32::from(*flags_index), Kind::String);
        }

        Instruction::ThrowNewTypeError { message }
        | Instruction::ThrowNewReferenceError { message } => {
            constant(errors, pc, op, code, u32::from(*message), Kind::String);
        }
        Instruction::ThrowMutateImmutable { index } => {
            constant(errors, pc, op, code, u32::from(*index), Kind::String);
        }

        Instruction::DefVar { binding_index }
        | Instruction::DefEvalVar { binding_index }
        | Instruction::GetLocator { binding_index } => {
            binding(errors, pc, op, u32::from(*binding_index), bindings_len);
        }
        Instruction::DefInitVar { src, binding_index }
        | Instruction::PutLexicalValue { src, binding_index }
        | Instruction::SetName { src, binding_index } => {
            reg(errors, *src);
            binding(errors, pc, op, u32::from(*binding_index), bindings_len);
        }
        Instruction::GetName { dst, binding_index }
        | Instruction::GetNameAndLocator { dst, binding_index }
        | Instruction::GetNameOrUndefined { dst, binding_index }
        | Instruction::DeleteName { dst, binding_index } => {
            reg(errors, *dst);
            binding(errors, pc, op, u32::from(*binding_index), bindings_len);
        }
        Instruction::GetNameGlobal {
            dst,
            binding_index,
            ic_index,
        } => {
            reg(errors, *dst);
            binding(errors, pc, op, u32::from(*binding_index), bindings_len);
            inline_cache(errors, pc, op, u32::from(*ic_index), ic_len);
        }
        Instruction::ThisForObjectEnvironmentName { dst, index } => {
            reg(errors, *dst);
            binding(errors, pc, op, u32::from(*index), bindings_len);
        }

        Instruction::GetMethod { object, name_index }
        | Instruction::DeletePropertyByName { object, name_index } => {
            reg(errors, *object);
            constant(errors, pc, op, code, u32::from(*name_index), Kind::String);
        }
        Instruction::DefineOwnPropertyByName {
            object,
            value,
            name_index,
        }
        | Instruction::DefineClassStaticMethodByName {
            object,
            value,
            name_index,
        }
        | Instruction::DefineClassMethodByName {
            object,
            value,
            name_index,
        }
        | Instruction::SetPropertyGetterByName {
            object,
            value,
            name_index,
        }
        | Instruction::DefineClassStaticGetterByName {
            object,
            value,
            name_index,
        }
        | Instruction::DefineClassGetterByName {
            object,
            value,
            name_index,
        }
        | Instruction::SetPropertySetterByName {
            object,
            value,
            name_index,
        }
        | Instruction::DefineClassStaticSetterByName {
            object,
            value,
            name_index,
        }
        | Instruction::DefineClassSetterByName {
            object,
            value,
            name_index,
        }
        | Instruction::SetPrivateField {
            object,
            value,
            name_index,
        }
        | Instruction::DefinePrivateField {
            object,
            value,
            name_index,
        }
        | Instruction::SetPrivateMethod {
            object,
            value,
            name_index,
        }
        | Instruction::SetPrivateSetter {
            object,
            value,
            name_index,
        }
        | Instruction::SetPrivateGetter {
            object,
            value,
            name_index,
        }
        | Instruction::PushClassFieldPrivate {
            object,
            value,
            name_index,
        }
        | Instruction::PushClassPrivateGetter {
            object,
            value,
            name_index,
        }
        | Instruction::PushClassPrivateSetter {
            object,
            value,
            name_index,
        } => {
            reg(errors, *object);
            reg(errors, *value);
            constant(errors, pc, op, code, u32::from(*name_index), Kind::String);
        }
        Instruction::GetPrivateField {
            dst,
            object,
            name_index,
        } => {
            reg(errors, *dst);
            reg(errors, *object);
            constant(errors, pc, op, code, u32::from(*name_index), Kind::String);
        }
        Instruction::PushClassPrivateMethod {
            object,
            proto,
            value,
            name_index,
        } => {
            reg(errors, *object);
            reg(errors, *proto);
            reg(errors, *value);
            constant(errors, pc, op, code, u32::from(*name_index), Kind::String);
        }

        Instruction::GetLengthProperty {
            dst,
            value,
            ic_index,
        }
        | Instruction::GetPropertyByName {
            dst,
            value,
            ic_index,
        } => {
            reg(errors, *dst);
            reg(errors, *value);
            inline_cache(errors, pc, op, u32::from(*ic_index), ic_len);
        }
        Instruction::GetPropertyByNameWithThis {
            dst,
            receiver,
            value,
            ic_index,
        } => {
            reg(errors, *dst);
            reg(errors, *receiver);
            reg(errors, *value);
            inline_cache(errors, pc, op, u32::from(*ic_index), ic_len);
        }
        Instruction::SetPropertyByName {
            value,
            object,
            ic_index,
        } => {
            reg(errors, *value);
            reg(errors, *object);
            inline_cache(errors, pc, op, u32::from(*ic_index), ic_len);
        }
        Instruction::SetPropertyByNameWithThis {
            value,
            receiver,
            object,
            ic_index,
        } => {
            reg(errors, *value);
            reg(errors, *receiver);
            reg(errors, *object);
            inline_cache(errors, pc, op, u32::from(*ic_index), ic_len);
        }

        Instruction::SetFunctionName { function, name, .. } => {
            reg(errors, *function);
            reg(errors, *name);
        }
        Instruction::ImportCall {
            specifier, options, ..
        } => {
            reg(errors, *specifier);
            reg(errors, *options);
        }
        Instruction::PushClassField {
            object,
            name,
            value,
            ..
        } => {
            reg(errors, *object);
            reg(errors, *name);
            reg(errors, *value);
        }
        Instruction::GetPropertyByIndex { dst, object, .. } => {
            reg(errors, *dst);
            reg(errors, *object);
        }
        Instruction::SetPropertyByIndex { value, object, .. } => {
            reg(errors, *value);
            reg(errors, *object);
        }

        Instruction::CallEval { scope_index, .. }
        | Instruction::CallEvalSpread { scope_index }
        | Instruction::PushScope { scope_index } => {
            constant(errors, pc, op, code, u32::from(*scope_index), Kind::Scope);
        }

        Instruction::TemplateCreate { dst, values, .. } => {
            reg(errors, *dst);
            for value in values {
                register_u32(errors, pc, op, register_count, *value);
            }
        }
        Instruction::PushPrivateEnvironment {
            class,
            name_indices,
        } => {
            reg(errors, *class);
            for index in name_indices {
                constant(errors, pc, op, code, *index, Kind::String);
            }
        }
        Instruction::ConcatToString { dst, values } => {
            reg(errors, *dst);
            for value in values {
                reg(errors, *value);
            }
        }
        Instruction::CopyDataProperties {
            object,
            source,
            excluded_keys,
        } => {
            reg(errors, *object);
            reg(errors, *source);
            for key in excluded_keys {
                reg(errors, *key);
            }
        }
    }
}

fn check_handlers(code: &CodeBlock, starts: &HashSet<u32>, errors: &mut Vec<String>) {
    for (i, handler) in code.handlers.iter().enumerate() {
        let start = handler.start.as_u32();
        let end = handler.end.as_u32();
        if start > end {
            errors.push(format!(
                "pc {start:06x}: handler {i}: start {start:06x} is after end {end:06x}"
            ));
        }
        if !starts.contains(&start) {
            errors.push(format!(
                "pc {start:06x}: handler {i}: start is not an instruction start"
            ));
        }
        if !starts.contains(&end) {
            errors.push(format!(
                "pc {end:06x}: handler {i}: end is not an instruction start"
            ));
        }
    }
}

fn check_globals(code: &CodeBlock, errors: &mut Vec<String>) {
    for (i, index) in code.global_lexs.iter().copied().enumerate() {
        constant_at(
            errors,
            &format!("global_lexs[{i}]"),
            code,
            index,
            Kind::String,
        );
    }
    for (i, index) in code.global_vars.iter().copied().enumerate() {
        constant_at(
            errors,
            &format!("global_vars[{i}]"),
            code,
            index,
            Kind::String,
        );
    }
    for (i, fun) in code.global_fns.iter().enumerate() {
        constant_at(
            errors,
            &format!("global_fns[{i}].name_index"),
            code,
            fun.name_index,
            Kind::String,
        );
        constant_at(
            errors,
            &format!("global_fns[{i}].function_index"),
            code,
            fun.function_index,
            Kind::Function,
        );
    }
}

/// Walk `code` and report every structural violation.
///
/// `Ok(())` means the bytecode decoded exactly to its end, every register,
/// jump, handler boundary, constant, binding and inline-cache operand is in
/// range, and each constant has the kind that opcode reads.
pub(crate) fn verify(code: &CodeBlock) -> Result<(), Vec<String>> {
    let mut errors = Vec::new();
    let decoded = decode(code, &mut errors);
    let starts = decoded.iter().map(|item| item.pc).collect::<HashSet<_>>();
    for item in &decoded {
        check_instruction(code, &starts, item, &mut errors);
    }
    check_handlers(code, &starts, &mut errors);
    check_globals(code, &mut errors);
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}
