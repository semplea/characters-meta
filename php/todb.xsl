<?xml version="1.0" encoding="utf-8"?>
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform" xmlns="http://www.tei-c.org/ns/1.0" xpath-default-namespace="http://www.w3.org/1999/xhtml">
	<xsl:output method="xml" encoding="utf-8" indent="no"/>
	<xsl:strip-space elements="*"/>

	<xsl:template match="html">
		<book>
			<xsl:apply-templates  disable-output-escaping="yes"/>
		</book>
	</xsl:template>

	<xsl:template match="head">
		<!-- nothing here -->
	</xsl:template>
	<xsl:template match="style">
		<!-- nothing here -->
	</xsl:template>

	<xsl:template match="body">
		<xsl:for-each select="div">
		<_>
			<xsl:apply-templates/>
		</_>
		<br/>
		</xsl:for-each>
	</xsl:template>
	<xsl:template match="h2">
		<br/>
		<h2><xsl:apply-templates/></h2>
		<br/>
	</xsl:template>
	<xsl:template match="p">
		<xsl:choose>
			<xsl:when test="contains(@class, 'MsoToc2')">
			</xsl:when>
			<xsl:otherwise>
				<p><xsl:apply-templates/></p>
			</xsl:otherwise>
		</xsl:choose>
	</xsl:template>
	<xsl:template match="i">
		<i><xsl:apply-templates/></i>
	</xsl:template>
	<xsl:template match="del">
	</xsl:template>


<!--
<xsl:template match="*/text()[normalize-space()]">
    <xsl:value-of select="normalize-space()"/>
</xsl:template>

<xsl:template match="*/text()[not(normalize-space())]" />
-->
</xsl:stylesheet>