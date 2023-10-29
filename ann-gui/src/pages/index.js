import { useState } from 'react';
import { Button, Checkbox, FormControl, FormLabel, Input, VStack, Modal, ModalOverlay, ModalContent, ModalHeader, ModalFooter, ModalBody, ModalCloseButton, NumberInput, NumberInputField, NumberInputStepper, NumberIncrementStepper, NumberDecrementStepper, Heading, Divider, Spacer, HStack } from "@chakra-ui/react";
import { CheckIcon } from "@chakra-ui/icons";
import axios from 'axios';
import SideMenu from './sideMenu';

export default function Home() {
  const [isOpen, setIsOpen] = useState(true);
  const [normalize, setNormalize] = useState(false);
  const [skipHeader, setSkipHeader] = useState(false);
  const [validationSplit, setValidationSplit] = useState(0.2);
  const [file, setFile] = useState(null);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleOk = async () => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('normalize', normalize);
    formData.append('skipHeader', skipHeader);
    formData.append('validationSplit', validationSplit);

    const response = await axios.post('http://localhost:5000/saveData', formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    });

    console.log({
      file,
      normalize,
      skipHeader,
      validationSplit
    });
    setIsOpen(false); 
  };

  return (
    <VStack spacing={4} padding={4} alignItems='flex-start'>
      <Modal isOpen={isOpen} onClose={() => setIsOpen(false)} size="xl">
        <ModalOverlay />
        <ModalContent padding={5} borderRadius="xl" boxShadow="2xl">
          <ModalHeader>
            <Heading size="lg" color="teal.500">Load the Dataset in CSV</Heading>
            <Divider my={3} borderColor="teal.300" />
          </ModalHeader>
          <ModalCloseButton />
          <ModalBody>
            <VStack spacing={5}>
              <FormControl>
                <Input type="file" size="lg" variant="flushed" focusBorderColor="teal.400" onChange={handleFileChange}/>
              </FormControl>
              <VStack spacing={2} alignItems="flex-start">
                <Checkbox size="lg" isChecked={normalize} onChange={(e) => setNormalize(e.target.checked)} colorScheme="teal" iconColor="white" icon={<CheckIcon boxSize="1.5em" />}>
                  Normalize
                </Checkbox>
                <Spacer />
                <Checkbox size="lg" isChecked={skipHeader} onChange={(e) => setSkipHeader(e.target.checked)} colorScheme="teal" iconColor="white" icon={<CheckIcon boxSize="1.5em" />}>
                  Skip Header
                </Checkbox>
              </VStack>
              <FormControl>
                <FormLabel> Validation Split</FormLabel>
                <NumberInput size="lg" value={validationSplit} onChange={(value) => setValidationSplit(value)} focusBorderColor="teal.400" min={0.0} max={0.9} step={0.1} precision={1}>
                  <NumberInputField />
                  <NumberInputStepper>
                    <NumberIncrementStepper />
                    <NumberDecrementStepper />
                  </NumberInputStepper>
                </NumberInput>
              </FormControl>
            </VStack>
          </ModalBody>
          <ModalFooter>
            <Button colorScheme="teal" size="lg" onClick={handleOk}>
              Ok
            </Button>
          </ModalFooter>
        </ModalContent>
      </Modal>
      <HStack>
        <SideMenu />
      </HStack>
    </VStack>
  );
}
