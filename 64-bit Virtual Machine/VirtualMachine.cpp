#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <cstdint>
#include <stdexcept>
#include <cstring> // For memcpy
#include <sstream> // For robust string parsing
#include <algorithm> // For std::replace

// Define the VM's instruction set (opcodes) using an enum.
// This provides a clear, human-readable way to represent machine instructions.
enum OpCode : uint8_t {
    HALT,       // 0: Stop execution
    LOADI,      // 1: Load an immediate (literal) 64-bit value into a register
    ADD,        // 2: Add two registers
    SUB,        // 3: Subtract two registers
    MUL,        // 4: Multiply two registers
    DIV,        // 5: Divide two registers
    PRINT,      // 6: Print a register's value to the console
    JUMP,       // 7: Unconditional jump to a memory address
    JUMP_EQ,    // 8: Conditional jump if a register's value is zero
};

// Represents the state of our virtual CPU, including its registers and program counter.
// Encapsulating these in a struct keeps the state organized.
struct CPUState {
    // Fixed number of registers (8) for simplicity.
    static const int NUM_REGISTERS = 8;
    // An array of 64-bit unsigned integers to simulate the registers.
    // The `uint64_t` type guarantees they are 64 bits wide regardless of the host system.
    uint64_t registers[NUM_REGISTERS] = {0};
    
    // The Program Counter (PC): points to the memory address of the next instruction.
    uint64_t pc = 0;
};

// Represents the entire Virtual Machine. This class contains the CPU state and memory.
class VirtualMachine {
public:
    // A simple representation of memory as a vector of bytes.
    static const size_t MEMORY_SIZE = 1024; // 1 KB of memory
    std::vector<uint8_t> memory;

    // The CPU state, which includes registers and PC.
    CPUState cpu;

    // A map to look up register indices by their string names (e.g., "R0").
    // This is used by the assembler to translate code.
    std::map<std::string, int> register_map = {
        {"R0", 0}, {"R1", 1}, {"R2", 2}, {"R3", 3},
        {"R4", 4}, {"R5", 5}, {"R6", 6}, {"R7", 7}
    };
    
    // Constructor to initialize memory and set up the VM
    VirtualMachine() : memory(MEMORY_SIZE, 0) {}

    // The main execution loop: this is the heart of the VM.
    void run() {
        // The fetch-decode-execute cycle runs until a HALT instruction is encountered.
        while (true) {
            // Fetch: Read the current instruction (opcode) from memory at the PC's address.
            uint8_t instruction = memory[cpu.pc];
            
            // Increment the program counter to point to the next byte, which may be an operand.
            cpu.pc++;

            // Decode and Execute: A switch statement efficiently decodes the instruction and executes the corresponding logic.
            switch (instruction) {
                case HALT: {
                    std::cout << "VM Halted." << std::endl;
                    return;
                }
                case LOADI: {
                    // Instruction format: [OpCode] [DestRegIndex] [Value (8 bytes)]
                    uint8_t dest_reg_idx = memory[cpu.pc];
                    cpu.pc++;
                    
                    uint64_t value;
                    // Copy the 8-byte value from memory into our 64-bit variable.
                    std::memcpy(&value, &memory[cpu.pc], sizeof(uint64_t));
                    cpu.registers[dest_reg_idx] = value;
                    cpu.pc += sizeof(uint64_t); // Advance PC past the 8-byte value.
                    break;
                }
                case ADD:
                case SUB:
                case MUL:
                case DIV: {
                    // Instruction format: [OpCode] [DestRegIndex] [Src1RegIndex] [Src2RegIndex]
                    uint8_t dest_reg_idx = memory[cpu.pc];
                    uint8_t src1_reg_idx = memory[cpu.pc + 1];
                    uint8_t src2_reg_idx = memory[cpu.pc + 2];
                    cpu.pc += 3; // Advance PC past the 3 register indices.

                    uint64_t src1 = cpu.registers[src1_reg_idx];
                    uint64_t src2 = cpu.registers[src2_reg_idx];
                    
                    switch (instruction) {
                        case ADD: cpu.registers[dest_reg_idx] = src1 + src2; break;
                        case SUB: cpu.registers[dest_reg_idx] = src1 - src2; break;
                        case MUL: cpu.registers[dest_reg_idx] = src1 * src2; break;
                        case DIV: 
                            if (src2 == 0) {
                                throw std::runtime_error("Division by zero!");
                            }
                            cpu.registers[dest_reg_idx] = src1 / src2; 
                            break;
                    }
                    break;
                }
                case PRINT: {
                    // Instruction format: [OpCode] [SrcRegIndex]
                    uint8_t src_reg_idx = memory[cpu.pc];
                    cpu.pc++;
                    std::cout << "R" << (int)src_reg_idx << ": " << cpu.registers[src_reg_idx] << std::endl;
                    break;
                }
                case JUMP: {
                    // Instruction format: [OpCode] [Address (8 bytes)]
                    uint64_t address;
                    std::memcpy(&address, &memory[cpu.pc], sizeof(uint64_t));
                    cpu.pc = address; // Set PC to the new address.
                    break;
                }
                case JUMP_EQ: {
                    // Instruction format: [OpCode] [SrcRegIndex] [Address (8 bytes)]
                    uint8_t src_reg_idx = memory[cpu.pc];
                    uint64_t address;
                    std::memcpy(&address, &memory[cpu.pc + 1], sizeof(uint64_t));
                    cpu.pc += 1 + sizeof(uint64_t);

                    if (cpu.registers[src_reg_idx] == 0) {
                        cpu.pc = address; // Conditional jump
                    }
                    break;
                }
                default: {
                    throw std::runtime_error("Unknown opcode: " + std::to_string(instruction));
                }
            }
        }
    }

    // A new method to execute a single, parsed instruction
    void execute_single_instruction(const std::string& line) {
        std::string clean_line = line;
        std::replace(clean_line.begin(), clean_line.end(), ',', ' ');
        std::stringstream ss(clean_line);
        std::string opcode_str;
        ss >> opcode_str;

        if (opcode_str == "LOADI") {
            std::string reg_name, value_str;
            ss >> reg_name >> value_str;
            if (register_map.count(reg_name)) {
                uint64_t value = std::stoull(value_str);
                cpu.registers[register_map.at(reg_name)] = value;
                std::cout << "Loaded " << value << " into " << reg_name << "." << std::endl;
            } else {
                throw std::runtime_error("Invalid register name: " + reg_name);
            }
        } else if (opcode_str == "ADD" || opcode_str == "SUB" || opcode_str == "MUL" || opcode_str == "DIV") {
            std::string dest_reg, src1_reg, src2_reg;
            ss >> dest_reg >> src1_reg >> src2_reg;
            if (register_map.count(dest_reg) && register_map.count(src1_reg) && register_map.count(src2_reg)) {
                uint64_t src1 = cpu.registers[register_map.at(src1_reg)];
                uint64_t src2 = cpu.registers[register_map.at(src2_reg)];
                uint64_t result;

                if (opcode_str == "ADD") {
                    result = src1 + src2;
                } else if (opcode_str == "SUB") {
                    result = src1 - src2;
                } else if (opcode_str == "MUL") {
                    result = src1 * src2;
                } else { // DIV
                    if (src2 == 0) {
                        throw std::runtime_error("Division by zero!");
                    }
                    result = src1 / src2;
                }
                cpu.registers[register_map.at(dest_reg)] = result;
                std::cout << "Operation successful. Result stored in " << dest_reg << "." << std::endl;
            } else {
                throw std::runtime_error("Invalid register names.");
            }
        } else if (opcode_str == "PRINT") {
            std::string reg_name;
            ss >> reg_name;
            if (register_map.count(reg_name)) {
                std::cout << "Register " << reg_name << ": " << cpu.registers[register_map.at(reg_name)] << std::endl;
            } else {
                throw std::runtime_error("Invalid register name: " + reg_name);
            }
        } else if (opcode_str == "JUMP" || opcode_str == "JUMP_EQ") {
            std::cout << "JUMP and JUMP_EQ instructions are not supported in interactive mode. Please use a multi-line program for these." << std::endl;
        } else {
            throw std::runtime_error("Unknown opcode: " + opcode_str);
        }
    }
};

// Simple assembler function to convert text-based instructions into byte code.
// This simplifies the process of creating a program for the VM.
// It uses a two-pass approach to handle forward references for jump labels.
std::vector<uint8_t> assemble(const std::vector<std::string>& program, VirtualMachine& vm) {
    std::vector<uint8_t> bytecode;
    std::map<std::string, uint64_t> labels;
    
    // Create a cleaned-up version of the program with all commas removed
    std::vector<std::string> clean_program = program;
    for (auto& line : clean_program) {
        std::replace(line.begin(), line.end(), ',', ' ');
    }

    // First pass: identify all labels and their corresponding memory addresses.
    // The address is simply the current size of the bytecode.
    size_t current_address = 0;
    for (const auto& line : clean_program) {
        std::stringstream ss(line);
        std::string opcode_str;
        ss >> opcode_str;

        if (opcode_str.back() == ':') {
            labels[opcode_str.substr(0, opcode_str.size() - 1)] = current_address;
        } else if (opcode_str == "LOADI") {
            current_address += 1 + 1 + 8;
        } else if (opcode_str == "ADD" || opcode_str == "SUB" || opcode_str == "MUL" || opcode_str == "DIV") {
            current_address += 1 + 3;
        } else if (opcode_str == "PRINT") {
            current_address += 1 + 1;
        } else if (opcode_str == "JUMP") {
            current_address += 1 + 8;
        } else if (opcode_str == "JUMP_EQ") {
            current_address += 1 + 1 + 8;
        } else if (opcode_str == "HALT") {
            current_address += 1;
        }
    }

    // Second pass: generate the actual bytecode, using the label map to fill in addresses.
    for (const auto& line : clean_program) {
        if (line.back() == ':') {
            continue; // Skip labels in the second pass
        } 
        
        std::stringstream ss(line);
        std::string opcode_str;
        ss >> opcode_str;

        if (opcode_str == "HALT") {
            bytecode.push_back(HALT);
        } else if (opcode_str == "LOADI") {
            bytecode.push_back(LOADI);
            std::string reg_name, value_str;
            ss >> reg_name >> value_str;
            uint64_t value = std::stoull(value_str);
            bytecode.push_back(vm.register_map.at(reg_name));
            const uint8_t* value_bytes = reinterpret_cast<const uint8_t*>(&value);
            for(size_t i = 0; i < sizeof(uint64_t); ++i) {
                bytecode.push_back(value_bytes[i]);
            }
        } else if (opcode_str == "ADD" || opcode_str == "SUB" || opcode_str == "MUL" || opcode_str == "DIV") {
            uint8_t opcode = (opcode_str == "ADD") ? ADD :
                             (opcode_str == "SUB") ? SUB :
                             (opcode_str == "MUL") ? MUL : DIV;
            bytecode.push_back(opcode);
            std::string dest_reg, src1_reg, src2_reg;
            ss >> dest_reg >> src1_reg >> src2_reg;
            bytecode.push_back(vm.register_map.at(dest_reg));
            bytecode.push_back(vm.register_map.at(src1_reg));
            bytecode.push_back(vm.register_map.at(src2_reg));
        } else if (opcode_str == "PRINT") {
            bytecode.push_back(PRINT);
            std::string reg_name;
            ss >> reg_name;
            bytecode.push_back(vm.register_map.at(reg_name));
        } else if (opcode_str == "JUMP") {
            bytecode.push_back(JUMP);
            std::string label_name;
            ss >> label_name;
            uint64_t address = labels.at(label_name);
            const uint8_t* address_bytes = reinterpret_cast<const uint8_t*>(&address);
            for(size_t i = 0; i < sizeof(uint64_t); ++i) {
                bytecode.push_back(address_bytes[i]);
            }
        } else if (opcode_str == "JUMP_EQ") {
            bytecode.push_back(JUMP_EQ);
            std::string reg_name, label_name;
            ss >> reg_name >> label_name;
            uint64_t address = labels.at(label_name);
            bytecode.push_back(vm.register_map.at(reg_name));
            const uint8_t* address_bytes = reinterpret_cast<const uint8_t*>(&address);
            for(size_t i = 0; i < sizeof(uint64_t); ++i) {
                bytecode.push_back(address_bytes[i]);
            }
        }
    }
    return bytecode;
}

// Main function to demonstrate the VM
int main() {
    VirtualMachine vm;

    std::cout << "----------------------------------------------------" << std::endl;
    std::cout << "Welcome to the Interactive Virtual Machine (REPL)." << std::endl;
    std::cout << "Enter an instruction or 'HALT' to exit." << std::endl;
    std::cout << "Example Instructions to try:" << std::endl;
    std::cout << "  LOADI R0, 100    // Load immediate value 100 into R0" << std::endl;
    std::cout << "  LOADI R1, 20     // Load immediate value 20 into R1" << std::endl;
    std::cout << "  ADD R2, R0, R1   // R2 = R0 + R1 (result is 120)" << std::endl;
    std::cout << "  SUB R3, R0, R1   // R3 = R0 - R1 (result is 80)" << std::endl;
    std::cout << "  MUL R4, R0, R1   // R4 = R0 * R1 (result is 2000)" << std::endl;
    std::cout << "  DIV R5, R0, R1   // R5 = R0 / R1 (result is 5)" << std::endl;
    std::cout << "  PRINT R2         // Print value of R2" << std::endl;
    std::cout << "  PRINT R3         // Print value of R3" << std::endl;
    std::cout << "  PRINT R4         // Print value of R4" << std::endl;
    std::cout << "  PRINT R5         // Print value of R5" << std::endl;
    std::cout << "----------------------------------------------------" << std::endl;

    std::string line;
    while (true) {
        std::cout << "> ";
        std::getline(std::cin, line);

        if (line.empty()) {
            continue;
        }

        // Check for the HALT command to exit the REPL loop cleanly
        if (line == "HALT") {
            std::cout << "VM Halted." << std::endl;
            break;
        }

        try {
            // Execute the single instruction directly
            vm.execute_single_instruction(line);
        } catch (const std::exception& e) {
            // Catch and display any errors during execution
            std::cerr << "Error: " << e.what() << std::endl;
        }
    }

    return 0;
}
