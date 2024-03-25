.. CFA Multisignal Renewal documentation master file, created by
   sphinx-quickstart on Sun Mar 17 15:31:47 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

CFA Multisignal Renewal Documentation
=====================================

Welcome to the CFA's `Multisignal Epidemiological Inference <https://github.com/CDCgov/multisignal-epi-inference>`_ (MSEI) project (*a.k.a.* signal fusion project), which consists of an internal forecasting model that leverages multiple data sources for enhancing epidemiological modeling of infectious disease outbreaks.

This repository is composed of two parts:

1.  **Model development** (`model folder <https://github.com/CDCgov/multisignal-epi-inference/blob/main/model>`_).
2.  **Analysis pipeline** (`pipeline folder <https://github.com/CDCgov/multisignal-epi-inference/blob/main/pipeline>`_).

Overview of the project follows:

.. mermaid::

   flowchart TD
      %% Main diagram
      io((P1: I/O\nDefinition)) --> |Dependency of| model((P2: Model\nPackage))
      io --> |Is used by| etl[[P3: ETL]]
      model --> |Is used in| run
      io -.-> |Possible\ndependency of|ww((Wastewater\nPackage))

      %% Definition of the pipe
      subgraph pipeline["Pipeline\n(Azure + GHA)"]
         etl --> |Feeds| run[["P4: Run the\nmodel"]]
      end
      run --> |Feeds| Outputs

      %% Definition of the outputs
      subgraph Outputs
         direction TB
         postp[[P5: Post\nProduction]]
         retro[[P6: Retrospective\nTesting]]
         bench[[P7: Benchmarking\n&A/B testing]]
      end

      %% Connections to the outputs
      io  --> |Is used by| Outputs
      postp --> manual[[Manual review]]
      manual --> share[[Share publicly]]


      %% Tagging sub-projects
      classDef tealNode fill:teal,color:white,stroke:white;
      class io,model,etl,run,postp,retro,bench,project,process tealNode;

Documentation Components
========================

.. toctree::
   :maxdepth: 1

   msei_reference/index
   test_reference/index
   state 
   faq
   usage
   

Indices And Tables
==================

.. toctree::
   :maxdepth: 1
 
   genindex
   modindex
   search
   glossary
   ctoc

Meta Information 
================

.. toctree::
   :maxdepth: 1
 
   contribute
   bugs
   license
   notices

.. todo:: Have docs folder exist in main, deploy using GHA. 

.. todo:: Add [State, Usage, Help, Tutorials/index, FAQ, HOWTOs/index] pages in Documentation Components. 

.. todo:: Add [About, History, Download, Copyright] pages in Meta Information. 

